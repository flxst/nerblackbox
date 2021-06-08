import json

import numpy as np
from transformers import AutoModelForTokenClassification
from argparse import Namespace
from torch.nn.functional import softmax
from typing import List, Union, Tuple, Any, Dict

from nerblackbox.modules.ner_training.ner_model import NerModel


class NerModelPredict(NerModel):
    """
    class that predicts tags for given text
    """

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
    ) -> "NerModelPredict":
        """load model in inference mode from checkpoint_path

        Args:
            checkpoint_path: path to checkpoint

        Returns:
            model loaded from checkpoint
        """
        model = super().load_from_checkpoint(
            checkpoint_path, map_location=None, tags_csv=None
        )
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
        :return: -
        """
        self.default_logger = None

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
            self.pretrained_model_name,
            num_labels=len(self.tag_list),
        )
        self.model.resize_token_embeddings(len(self.tokenizer))  # due to addtional_special_tokens = NEWLINE_TOKENS

    ####################################################################################################################
    # PREDICT
    ####################################################################################################################
    def predict(self, examples: Union[str, List[str]]) -> List[Namespace]:
        """predict tags

        Args:
            examples: e.g. ["example 1", "example 2"]

        Returns:
            predictions: with .internal [list] of (word, tag) tuples \
                         and  .external [list] of (word, tag) tuples
        """
        return self._predict(examples, proba=False)

    def predict_proba(self, examples: Union[str, List[str]]) -> List[Namespace]:
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
            predictions: with .internal [list] of (word, tag / proba_dist) tuples
                         and  .external [list] of dict w/ keys = char_start, char_end, word, tag / proba_dist
        """
        if isinstance(examples, str):
            examples = [examples]

        predictions = (
            list()
        )
        for example in examples:
            predict_dataloader = self._get_predict_dataloader([example])

            if self.data_preprocessor.do_lower_case:
                example = example.lower()

            example_tokens = (
                list()
            )
            example_token_predictions = (
                list()
            )
            ######################################
            # 1 example, individual chunks, tokens
            ######################################
            for sample in predict_dataloader:
                token_tensors, tokens = self._predict_on_tokens(sample)
                if proba is False:
                    token_predictions = self._turn_tensors_into_tags(
                        token_tensors
                    )
                else:
                    token_predictions = (
                        self._turn_tensors_into_tag_probability_distributions(
                            token_tensors
                        )
                    )
                example_tokens.extend(tokens)
                example_token_predictions.extend(token_predictions)

            ######################################
            # 1 example, merged chunks, tokens -> words
            ######################################
            example_word_predictions = self._get_tags_on_words_between_special_tokens(
                example_tokens, example_token_predictions
            )

            example_word_predictions["external"] = self._restore_unknown_tokens(example_word_predictions["external"],
                                                                                example)

            prediction = self._namespace_prediction(
                example_word_predictions,
            )

            predictions.append(prediction)

        ######################################
        # all examples, merged chunks, words
        ######################################
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
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
            softmax(output_token_tensors[i], dim=0)
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
    def _get_tags_on_words_between_special_tokens(tokens: List[str],
                                                  example_token_predictions: List[Any]):
        """
        Args:
            tokens:                    [list] of [str]
            example_token_predictions  [list] of [str] or [prob dist]

        Returns:
            example_word_predictions   [list] of [str] or [prob dist]
        """
        # predict tags on words between [CLS] and [SEP]
        example_word_predictions = {
            "internal": list(),
            "external": list(),
        }
        for token, example_token_prediction in zip(tokens, example_token_predictions):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                example_word_predictions["internal"].append([token, example_token_prediction])
                if not token.startswith("##"):
                    example_word_predictions["external"].append([token, example_token_prediction])
                else:
                    example_word_predictions["external"][-1][0] += token.strip("##")

        for field in ["internal", "external"]:
            example_word_predictions[field] = [tuple(elem) for elem in example_word_predictions[field]]

        return example_word_predictions

    @staticmethod
    def _restore_unknown_tokens(example_word_predictions_external: List[Tuple[str, str]],
                                _example: str,
                                verbose: bool = True) -> List[Dict[str, str]]:
        """
        - replace "[UNK]" tokens by the original token
        - enrich tokens with char_start & char_end

        Args:
            example_word_predictions_external: e.g. [('example', 'O'), ('of', 'O), ('author', 'PERSON'), ..]
            _example: 'example of author'

        Returns:
            example_word_predictions_external: e.g. [('example', 'O'), ('of', 'O), ('author', 'PERSON'), ..]
        """
        _predictions_external = list()

        # 1. get margins of known tokens
        token_char_margins = list()
        char_start = 0
        for token, _ in example_word_predictions_external:
            if token != "[UNK]":
                try:
                    char_start = _example.index(token, char_start)
                    token_char_margins.append((char_start, char_start + len(token)))
                    char_start += len(token)
                except ValueError:
                    print(f"! token = {token} not found in example[{char_start}:]")
                    token_char_margins.append((None, None))
            else:
                token_char_margins.append((None, None))

        # 2. restore unknown tokens
        for i, (token, tag) in enumerate(example_word_predictions_external):
            if token_char_margins[i][0] is not None and token_char_margins[i][1] is not None:
                _predictions_external.append(
                    {
                        "char_start": str(token_char_margins[i][0]),
                        "char_end": str(token_char_margins[i][1]),
                        "token": token,
                        "tag": tag,
                    }
                )
            else:
                for j in range(2):
                    assert token_char_margins[i][j] is None, \
                        f"ERROR! token_char_margin[{i}][{j}] is not None for token = {token}"

                char_start_margin, char_end_margin = None, None
                for k in range(1, 10):
                    if i-k < 0:
                        char_start_margin = 0
                        break
                    elif token_char_margins[i-k][1] is not None:
                        char_start_margin = token_char_margins[i-k][1]
                        break
                for k in range(1, 10):
                    if i+k >= len(token_char_margins):
                        char_end_margin = len(_example)
                        break
                    elif token_char_margins[i+k][0] is not None:
                        char_end_margin = token_char_margins[i+k][0]
                        break
                assert char_start_margin is not None, f"ERROR! could not find char_start_margin"
                assert char_end_margin is not None, f"ERROR! could not find char_end_margin"

                new_token = _example[char_start_margin: char_end_margin].strip()

                if len(new_token):
                    char_start = _example.index(new_token, char_start_margin)
                    char_end = char_start + len(new_token)
                    if verbose:
                        print(f"! restored unknown token = {new_token} between "
                              f"char_start = {char_start_margin}, "
                              f"char_end = {char_end_margin}"
                              )
                    _predictions_external.append(
                        {
                            "char_start": str(char_start),
                            "char_end": str(char_end),
                            "token": new_token,
                            "tag": tag,
                        }
                    )
                else:
                    print(f"! dropped unknown empty token between "
                          f"char_start = {char_start_margin}, "
                          f"char_end = {char_end_margin}"
                          )

        return _predictions_external

    @staticmethod
    def _namespace_prediction(example_word_predictions):
        """
        :param example_word_predictions   [dict] w/ keys = "internal", "external"
                                                   & values = [list] of [str] or [prob dist]
        :return: prediction              [Namespace] w/ attributes = 'internal' | 'external'
                                                     & values = [list] of [tuples] (word, tag / tag prob dist)
        """
        prediction = Namespace(
            **example_word_predictions
        )
        return prediction
