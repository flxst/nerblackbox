
import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.base_formatter import BaseFormatter


class SUCFormatter(BaseFormatter):

    def __init__(self):
        label_list = ['PER', 'ORG', 'LOC', 'OBJ', 'WRK']
        super().__init__(label_list)

    def modify_ner_label_mapping(self, ner_label_mapping_original, with_tags: bool):
        """
        customize ner label mapping if wanted
        -------------------------------------
        :param ner_label_mapping_original: [dict] w/ keys = labels in original data, values = labels in original data
        :param with_tags: [bool], if True: create labels with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple labels, e.g. 'PER', 'LOC', ..
        :return: ner_label_mapping: [dict] w/ keys = labels in original data, values = labels in formatted data
        """
        ner_label_mapping = ner_label_mapping_original
        return ner_label_mapping
