
import pandas as pd

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
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
    def get_data(self, verbose: bool):
        """
        I: get data
        -----------
        :param verbose: [bool]
        :return: -
        """
        print('(SUC: nothing to do)')

    def modify_ner_tag_mapping(self, ner_tag_mapping_original, with_tags: bool):
        """
        II: customize ner tag mapping if wanted
        -------------------------------------
        :param ner_tag_mapping_original: [dict] w/ keys = tags in original data, values = tags in original data
        :param with_tags: [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple tags, e.g. 'PER', 'LOC', ..
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        ner_tag_mapping = ner_tag_mapping_original
        return ner_tag_mapping

    def format_data(self, valid_fraction: float):
        """
        III: format data
        ----------------
        :param valid_fraction: [float]
        :return: -
        """
        # valid, test -> train
        csvs = self._read_csvs(['valid', 'test'])
        self._write_csvs_train(csvs)

        # train       -> valid (& test)
        csvs = self._read_csvs(['train'])
        self._write_csvs_valid_test(csvs, valid_fraction)

    ####################################################################################################################
    # HELPER: CSVS
    ####################################################################################################################
    def _read_csvs(self, phases):
        csv_phases = [self._read_original_file(phase) for phase in phases]
        return pd.concat(csv_phases, ignore_index=True)

    def _write_csvs_train(self, _csvs):
        self._write_formatted_csv('train', _csvs)

    def _write_csvs_valid_test(self, _csvs, _valid_fraction):
        split_index = int(len(_csvs) * _valid_fraction)
        csvs_valid = _csvs.iloc[:split_index]
        csvs_test = _csvs.iloc[split_index:]
        self._write_formatted_csv('valid', csvs_valid)
        self._write_formatted_csv('test', csvs_test)

    ####################################################################################################################
    # HELPER: READ / WRITE
    ####################################################################################################################
    @staticmethod
    def _read_original_file(phase):
        file_path = f'datasets/ner/SUC/{phase}_original.csv'
        return pd.read_csv(file_path, sep='\t', header=None)

    @staticmethod
    def _write_formatted_csv(phase, csvs):
        file_path = f'datasets/ner/SUC/{phase}.csv'
        csvs.to_csv(file_path, sep='\t', index=False, header=None)
