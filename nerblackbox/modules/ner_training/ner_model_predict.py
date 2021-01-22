import json
import numpy as np
from transformers import AutoModelForTokenClassification
from argparse import Namespace
from torch.nn.functional import softmax
from typing import List, Union

from nerblackbox.modules.ner_training.ner_model import NerModel


class NerModelPredict(NerModel):
    """
    class that predicts tags for given text
    """

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location=None,
        tags_csv=None,
    ) -> "NerModelPredict":
        """load model in inference mode from checkpoint_path

        Args:
            checkpoint_path: path to checkpoint

        Returns:
            model loaded from checkpoint
        """
        model = super().load_from_checkpoint(checkpoint_path, map_location, tags_csv)
        model.freeze()  # for inference mode
        return model

    def __init__(self, hparams: Namespace):
        """
        Args:
            hparams: attr experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__(hparams)

    ####################################################################################################################
    # Abstract Base Methods ############################################################################################
    ####################################################################################################################
    def _preparations(self):
        """
        :created attr: default_logger    [DefaultLogger]
        :created attr: logged_metrics    [LoggedMetrics]
        :created attr: tokenizer         [transformers AutoTokenizer]
        :created attr: data_preprocessor [DataPreprocessor]
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]
        :return: -
        """
        # predict
        self._preparations_predict()  # attr: default_logger
        self._preparations_data_general()  # attr: tokenizer, data_preprocessor
        self._preparations_data_predict()  # attr: tag_list, model

    def _preparations_predict(self):
        """
        :created attr: default_logger         [None]
        :created attr: pretrained_model_name  [str]
        :return: -
        """
        self.default_logger = None
        self.pretrained_model_name = self.params.pretrained_model_name

    def _preparations_data_predict(self):
        """
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]
        :return: -
        """
        # tag_list
        self.tag_list = json.loads(self.hparams.tag_list)

        # model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.params.pretrained_model_name, num_labels=len(self.tag_list)
        )

    ####################################################################################################################
    # PREDICT
    ####################################################################################################################
    def predict(self, examples: List[str]) -> List[Namespace]:
        """predict tags

        Args:
            examples: e.g. ["example 1", "example 2"]

        Returns:
            predictions: with .internal [list] of (word, tag) tuples \
                         and  .external [list] of (word, tag) tuples
        """
        return self._predict(examples, proba=False)

    def predict_proba(self, examples: List[str]) -> List[Namespace]:
        """predict probabilities for tags

        Args:
            examples: e.g. ["example 1", "example 2"]
        Returns:
            predictions: with .internal [list] of (word, proba_dist) tuples \
                         and  .external [list] of (word, proba_dist) tuples \
                         where proba_dist = [dict] that maps self.tag_list to probabilities
        """
        return self._predict(examples, proba=True)

    def _predict(
        self, examples: Union[str, List[str]], proba: bool = False
    ) -> List[Namespace]:
        """predict tags or probabilities for tags

        Args:
            examples: e.g. ["example 1", "example 2"]
            proba: predict probabilities instead of labels

        Returns:
            predictions: with .internal [list] of (word, tag / proba_dist) tuples \
                         and  .external [list] of (word, tag / proba_dist) tuples
        """
        if isinstance(examples, str):
            examples = [examples]

        predict_dataloader = self._get_predict_dataloader(examples)
        examples_tokenized = self._get_tokenized_examples(
            examples
        )  # for external predictions

        # get predictions
        predictions = (
            list()
        )  # for each example: .internal/.external = list of tuples (word, tag)
        for example_tokenized, sample in zip(examples_tokenized, predict_dataloader):
            output_token_tensors, tokens = self._predict_on_tokens(sample)
            if proba is False:
                output_token_predictions = self._turn_tensors_into_tags(
                    output_token_tensors
                )
            else:
                output_token_predictions = (
                    self._turn_tensors_into_tag_probability_distributions(
                        output_token_tensors
                    )
                )
            output_word_predictions = self._get_tags_on_words_between_special_tokens(
                tokens, output_token_predictions
            )
            prediction = self._summarize_prediction(
                output_word_predictions, example_tokenized
            )
            predictions.append(prediction)
        return predictions

    ####################################################################################################################
    # PREDICT HELPER METHODS
    ####################################################################################################################
    def _get_predict_dataloader(self, examples):
        """
        :param examples:            [list] of [str]
        :return: predict_dataloader [torch dataloader]
        """
        # input_examples
        input_examples = self.data_preprocessor.get_input_examples_predict(
            examples=examples,
        )

        # dataloader
        dataloader = self.data_preprocessor.to_dataloader(
            input_examples, self.tag_list, batch_size=1
        )

        return dataloader["predict"]

    def _get_tokenized_examples(self, examples):
        """
        :param examples:            [list] of [str],           e.g. ['Ett exempel', 'något annat']
        :return: examples_tokenized [list] of [list] of [str], e.g. [['Ett', 'exempel'], ['något', 'annat']]
        """
        examples_tokenized = [
            self.tokenizer.basic_tokenizer.tokenize(example) for example in examples
        ]
        return examples_tokenized

    def _predict_on_tokens(self, sample):
        """
        :param sample: [list] w/ 4 tensors: input_ids, attention_mask, segment_ids, label_ids
        :return: output_token_tensors [list] of [torch tensor] of shape [1, #tags]
        :return: tokens               [list] of [str]
        """
        (
            input_ids,  # shape: [1, seq_length]
            attention_mask,  # shape: [1, seq_length]
            segment_ids,  # shape: [1, seq_length]
            label_ids,  # shape: [1, seq_length]
        ) = sample

        output = self.model(
            input_ids, attention_mask, segment_ids, label_ids
        )  # shape: [1 (=#examples), 1 (=#batch_size), seq_length, #tags]
        output_token_tensors = [
            output[0][0][i]  # .detach().numpy()
            for i in range(self._hparams.max_seq_length)
        ]  # shape: [seq_length, #tags]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return output_token_tensors, tokens

    def _turn_tensors_into_tags(self, output_token_tensors):
        """
        :param output_token_tensors       [list] of [torch tensor] of shape [1, #tags]
        :return: output_token_predictions [list] of [str]
        """
        return [
            self.tag_list[np.argmax(output_token_tensors[i].detach().numpy())]
            for i in range(self._hparams.max_seq_length)
        ]

    def _turn_tensors_into_tag_probability_distributions(self, output_token_tensors):
        """
        :param output_token_tensors       [list] of [torch tensor] of shape [1, #tags]
        :return: output_token_predictions [list] of [prob dist], i.e. dict that maps tags to probabilities
        """

        probability_distributions = [
            softmax(output_token_tensors[i])
            for i in range(self._hparams.max_seq_length)
        ]

        tag_probability_distribution = [
            {
                self.tag_list[j]: float(
                    probability_distributions[i][j].detach().numpy()
                )
                for j in range(len(self.tag_list))
            }
            for i in range(self._hparams.max_seq_length)
        ]

        return tag_probability_distribution

    @staticmethod
    def _get_tags_on_words_between_special_tokens(tokens, output_token_predictions):
        """
        :param tokens:                    [list] of [str]
        :param output_token_predictions   [list] of [str] or [prob dist]
        :return: output_word_predictions  [list] of [str] or [prob dist]
        """
        # predict tags on words between [CLS] and [SEP]
        output_word_predictions = list()
        for token, output_token_prediction in zip(tokens, output_token_predictions):
            if token == "[SEP]":
                break
            if token != "[CLS]":
                if not token.startswith("##"):
                    output_word_predictions.append([token, output_token_prediction])
                else:
                    output_word_predictions[-1][0] += token.strip("##")

        return output_word_predictions

    @staticmethod
    def _summarize_prediction(output_word_predictions, example_tokenized):
        """
        :param output_word_predictions   [list] of [str] or [prob dist]
        :param example_tokenized         [list] of [str], e.g. ['Ett', 'exempel']
        :return: prediction              [Namespace] w/ attributes = 'internal' | 'external'
                                                     & values = [list] of [tuples] (word, tag / tag prob dist)
        """
        prediction_internal = [tuple(elem) for elem in output_word_predictions]
        prediction_external = [
            tuple([token, elem[1]])
            for token, elem in zip(example_tokenized, output_word_predictions)
        ]
        prediction = Namespace(
            **{"internal": prediction_internal, "external": prediction_external}
        )
        return prediction
