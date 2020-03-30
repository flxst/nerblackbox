
import shutil

from os.path import abspath, dirname, join
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

    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        for phase in ['train', 'val', 'test']:
            original_file_path = join(self.dataset_path, f'{phase}_original.csv')
            formatted_file_path = join(self.dataset_path, f'{phase}_formatted.csv')
            shutil.copy2(original_file_path, formatted_file_path)

    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float]
        :return: -
        """
        # train -> train
        df_train = self._read_formatted_files(['train'])
        self._write_final_csv('train', df_train)

        # val  -> val
        df_val = self._read_formatted_files(['val'])
        self._write_final_csv('val', df_val)

        # test  -> test
        df_test = self._read_formatted_files(['test'])
        self._write_final_csv('test', df_test)

        """
        # val, test -> train
        df_train = self._read_formatted_files(['val', 'test'])
        self._write_final_csv('train', df_train)

        # train       -> val & test
        df_val_test = self._read_formatted_files(['train'])
        df_val, df_test = self._split_val_test(df_val_test, val_fraction)
        self._write_final_csv('val', df_val)
        self._write_final_csv('test', df_test)
        """
