
import pandas as pd

import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.base_formatter import BaseFormatter


class SUCFormatter(BaseFormatter):

    def __init__(self):
        ner_dataset = 'SUC'
        ner_tag_list = ['PER', 'ORG', 'LOC', 'OBJ', 'WRK']
        super().__init__(ner_dataset, ner_tag_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def modify_ner_tag_mapping(self, ner_tag_mapping_original, with_tags: bool):
        """
        customize ner tag mapping if wanted
        -------------------------------------
        :param ner_tag_mapping_original: [dict] w/ keys = tags in original data, values = tags in original data
        :param with_tags: [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple tags, e.g. 'PER', 'LOC', ..
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        ner_tag_mapping = ner_tag_mapping_original
        return ner_tag_mapping

    def read_original_file(self, phase):
        file_path = f'datasets/ner/SUC/{phase}_original.csv'
        return pd.read_csv(file_path, sep='\t', header=None)

    def write_formatted_csv(self, phase, csvs, dataset_path):
        file_path = f'datasets/ner/SUC/{phase}.csv'
        csvs.to_csv(file_path, sep='\t', index=False, header=None)

