
import json
from transformers import AutoModelForTokenClassification

from ner_black_box.ner_training.lightning_ner_model import LightningNerModel


class LightningNerModelPredict(LightningNerModel):

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
