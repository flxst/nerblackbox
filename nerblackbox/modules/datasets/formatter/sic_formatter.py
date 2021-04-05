import os
import subprocess
from os.path import join

from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter
from nerblackbox.modules.utils.env_variable import env_variable


class SICFormatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "sic"
        ner_tag_list = [
            "person",
            "animal",
            "myth",
            "place",
            "inst",
            "product",
            "work",
            "event",
            "other",
        ]
        super().__init__(ner_dataset, ner_tag_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def get_data(self, verbose: bool):
        """
        I: get data
        ----------
        :param verbose: [bool]
        :return: -
        """
        bash_cmds = [
            f"mkdir {env_variable('DIR_DATASETS')}/_sic",
            f"curl -o {env_variable('DIR_DATASETS')}/_sic/sic.zip "
            "https://www.ling.su.se/polopoly_fs/1.99145.1380811903\!/menu/standard/file/sic.zip",
            f"cd {env_variable('DIR_DATASETS')}/_sic && unzip -o sic.zip",
            f"mkdir {env_variable('DIR_DATASETS')}/sic/raw_data",
            f"mv {env_variable('DIR_DATASETS')}/_sic/sic/annotated/* {env_variable('DIR_DATASETS')}/sic/raw_data",
            f"rm -r {env_variable('DIR_DATASETS')}/_sic",
            f"cat {env_variable('DIR_DATASETS')}/sic/raw_data/*.conll "
            f"> {env_variable('DIR_DATASETS')}/sic/sic-train.conll",
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
        ------------------------------------------------
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return dict()

    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        for phase in ["train"]:
            rows = self._read_original_file(phase)
            self._write_formatted_csv(phase, rows)

    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float]
        :return: -
        """
        # train -> train, val, test
        df_train_val_test = self._read_formatted_csvs(["train"])
        df_train_val, df_test = self._split_off_validation_set(
            df_train_val_test, val_fraction
        )
        df_train, df_val = self._split_off_validation_set(df_train_val, val_fraction)
        self._write_final_csv("train", df_train)
        self._write_final_csv("val", df_val)
        self._write_final_csv("test", df_test)

    ####################################################################################################################
    # HELPER: READ ORIGINAL
    ####################################################################################################################
    def _read_original_file(self, phase):
        """
        III: format data
        ---------------------------------------------
        :param phase:   [str] 'train' or 'test'
        :return: _rows: [list] of [list] of [str], e.g. [[], ['Inger', 'PER'], ['säger', '0'], ..]
        """
        file_name = {
            "train": "sic-train.conll",
        }
        file_path_original = join(self.dataset_path, file_name[phase])

        _rows = list()
        if os.path.isfile(file_path_original):
            with open(file_path_original) as f:
                for i, row in enumerate(f.readlines()):
                    _rows.append(row.strip().split())
            print(f"\n> read {file_path_original}")
        else:
            raise Exception(f"> original file {file_path_original} could not be found.")

        _rows = [
            [row[1], self.transform_tags(row[-3], row[-2])]
            if len(row) > 0
            and row[-3]
            not in ["_", "U"]  # filter out "_-_" and "_-person" (bugs in SIC dataset)
            else list()
            for row in _rows
        ]

        return _rows

    @staticmethod
    def transform_tags(bio, tag):
        """
        :param bio: [str] 'O', 'B', 'I'
        :param tag: [str] '_', 'person', ..
        :return: transformed tag: [str], e.g. 'O', 'B-person', ..
        """
        if bio == "O":
            return "O"
        else:
            return f"{bio}-{tag}"
