
import os
import subprocess

from os.path import abspath, dirname, join
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.base_formatter import BaseFormatter


class SwedishNerCorpusFormatter(BaseFormatter):

    def __init__(self):
        ner_dataset = 'swedish_ner_corpus'
        ner_tag_list = ['PER', 'ORG', 'LOC', 'MISC', 'PRG', 'ORG*']
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
        bash_cmds = [
            'git clone https://github.com/klintan/swedish-ner-corpus.git datasets/ner/swedish_ner_corpus',
            'echo \"*\" > datasets/ner/swedish_ner_corpus/.gitignore',
        ]

        for bash_cmd in bash_cmds:
            if verbose:
                print(bash_cmd)

            try:
                subprocess.run(bash_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e)

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

        # take care of extra case: ORG*
        if with_tags:
            ner_tag_mapping['B-ORG*'] = 'B-ORG'
            ner_tag_mapping['I-ORG*'] = 'I-ORG'
            ner_tag_mapping['B-PRG'] = 'O'
            ner_tag_mapping['I-PRG'] = 'O'
        else:
            ner_tag_mapping['ORG*'] = 'ORG'
            ner_tag_mapping['PRG'] = 'O'

        return ner_tag_mapping

    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        for phase in ['train', 'test']:
            rows = self._read_original_file(phase)
            self._write_formatted_csv(phase, rows)

    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float]
        :return: -
        """
        # train -> train, val
        df_train_val = self._read_formatted_files(['train'])
        df_train, df_val = self._split_off_validation_set(df_train_val, val_fraction)
        self._write_final_csv('train', df_train)
        self._write_final_csv('val', df_val)

        # test  -> test
        df_test = self._read_formatted_files(['test'])
        self._write_final_csv('test', df_test)

    ####################################################################################################################
    # HELPER: READ ORIGINAL
    ####################################################################################################################
    def _read_original_file(self, phase):
        """
        - read original text files
        ---------------------------------------------
        :param phase:   [str] 'train' or 'test'
        :return: _rows: [list] of [list] of [str], e.g. [['Inger', 'PER'], ['säger', '0'], ..]
        """
        file_path_original = join(self.dataset_path, f'{phase}_corpus.txt')

        _rows = list()
        if os.path.isfile(file_path_original):
            with open(file_path_original) as f:
                for row in f.readlines():
                    _rows.append(row.strip().split())
            print(f'\n> read {file_path_original}')

        return _rows
