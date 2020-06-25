
import json
import numpy as np
from transformers import AutoModelForTokenClassification
from argparse import Namespace

from nerblackbox.modules.ner_training.lightning_ner_model import LightningNerModel


class LightningNerModelPredict(LightningNerModel):

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint_path,
                             map_location=None,
                             tags_csv=None,
                             ):
        model = super().load_from_checkpoint(checkpoint_path, map_location, tags_csv)
        model.freeze()  # for inference mode
        return model

    def __init__(self, hparams):
        """
        :param hparams: [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__(hparams)

    ####################################################################################################################
    # PREPARATIONS
    ####################################################################################################################
    def _preparations(self):
        # predict
        self._preparations_predict()       # attr: default_logger
        self._preparations_data_general()  # attr: tokenizer, data_preprocessor
        self._preparations_data_predict()  # attr: tag_list, model

    def _preparations_predict(self):
        """
        :created attr: default_logger    [None]
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
        self.model = AutoModelForTokenClassification.from_pretrained(self.params.pretrained_model_name,
                                                                     num_labels=len(self.tag_list))

    ####################################################################################################################
    # PREDICT
    ####################################################################################################################
    def predict(self, examples):
        """
        :param examples:     [list] of [str]
        :return: predictions [list] of [Namespace] with .internal [list] of (word, tag) tuples
                                                   and  .external [list] of (word, tag) tuples
        """
        if isinstance(examples, str):
            examples = [examples]

        # input_examples
        input_examples = self.data_preprocessor.get_input_examples_predict(
            examples=examples,
        )

        examples_tokenized = [self.tokenizer.basic_tokenizer.tokenize(example)
                              for example in examples]
        # dataloader
        dataloader = self.data_preprocessor.to_dataloader(input_examples,
                                                          self.tag_list,
                                                          batch_size=1)

        # get predictions
        predictions = list()  # for each example: .internal/.external = list of tuples (word, tag)
        for example_tokenized, sample in zip(examples_tokenized, dataloader['predict']):
            # predict tags on tokens
            input_ids, attention_mask, segment_ids, label_ids, = sample
            output_tensors = self.model(input_ids, attention_mask, segment_ids, label_ids)
            output_token_tags = [self.tag_list[np.argmax(output_tensors[0][0][i].detach().numpy())]
                                 for i in range(self._hparams.max_seq_length)]

            # predict tags on words between [CLS] and [SEP]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            output_word_tags = list()
            for token, output_token_tag in zip(tokens, output_token_tags):
                if token == '[SEP]':
                    break
                if token != '[CLS]':
                    if not token.startswith('##'):
                        output_word_tags.append([token, output_token_tag])
                    else:
                        output_word_tags[-1][0] += token.strip('##')

            prediction_internal = [tuple(elem) for elem in output_word_tags]
            prediction_external = [tuple([token, elem[1]])
                                   for token, elem in zip(example_tokenized, output_word_tags)]
            prediction = Namespace(**{'internal': prediction_internal,
                                      'external': prediction_external})
            predictions.append(prediction)

        return predictions
