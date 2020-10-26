import os
import subprocess

from os.path import join

from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter
from nerblackbox.modules.utils.env_variable import env_variable


class SwedishNerCorpusFormatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "swedish_ner_corpus"
        ner_tag_list = ["PER", "ORG", "LOC", "MISC", "PRG", "ORG*"]
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
            f'git clone https://github.com/klintan/swedish-ner-corpus.git {env_variable("DIR_DATASETS")}/_swedish_ner_corpus',
            f'mv {env_variable("DIR_DATASETS")}/_swedish_ner_corpus/* {env_variable("DIR_DATASETS")}/swedish_ner_corpus',
            f'rm -rf {env_variable("DIR_DATASETS")}/_swedish_ner_corpus',
        ]

        for bash_cmd in bash_cmds:
            if verbose:
                print(bash_cmd)

            try:
                subprocess.run(bash_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e)

    def create_ner_tag_mapping(self):
        """
        II: customize ner_training tag mapping if wanted
        -------------------------------------
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return {
            "ORG*": "ORG",
            "PRG": "O",
        }

    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        for phase in ["train", "test"]:
            rows = self._read_original_file(phase)
            self._write_formatted_csv(phase, rows)

    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float], e.g. 0.3
        :return: -
        """
        # train -> train, val
        df_train_val = self._read_formatted_csvs(["train"])
        df_train, df_val = self._split_off_validation_set(df_train_val, val_fraction)
        self._write_final_csv("train", df_train)
        self._write_final_csv("val", df_val)

        # test  -> test
        df_test = self._read_formatted_csvs(["test"])
        self._write_final_csv("test", df_test)

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
        file_path_original = join(self.dataset_path, f"{phase}_corpus.txt")

        _rows = list()
        if os.path.isfile(file_path_original):
            with open(file_path_original) as f:
                for row in f.readlines():
                    _rows.append(row.strip().split())
            print(f"\n> read {file_path_original}")

        _rows = [row if len(row) == 2 else list() for row in _rows]

        return _rows
