import os
import subprocess
from os.path import join

from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter
from nerblackbox.modules.utils.env_variable import env_variable


class SweNercFormatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "swe_nerc"
        ner_tag_list = ["PRS", "LOC", "GRO", "EVN", "TME", "WRK", "SMP", "MNT"]
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
            f"mkdir {env_variable('DIR_DATASETS')}/_swe_nerc",
            f"curl -o {env_variable('DIR_DATASETS')}/_swe_nerc/swe_nerc.tar.gz "
            "https://spraakbanken.gu.se/lb/resurser/swe-nerc/Swe-NERC-v1.0.tar.gz",
            f"cd {env_variable('DIR_DATASETS')}/_swe_nerc && tar -xzf swe_nerc.tar.gz",
            f"mkdir {env_variable('DIR_DATASETS')}/swe_nerc/raw_data",
            f"mv {env_variable('DIR_DATASETS')}/_swe_nerc/Swe-NERC-v1.0/manually-tagged-part/*.tsv {env_variable('DIR_DATASETS')}/swe_nerc/raw_data",
            f"rm -r {env_variable('DIR_DATASETS')}/_swe_nerc",
            f"echo '\t\t' | tee -a {env_variable('DIR_DATASETS')}/swe_nerc/raw_data/*.tsv",
            #####
            f"cat {env_variable('DIR_DATASETS')}/swe_nerc/raw_data/*-01.tsv "
            f"> {env_variable('DIR_DATASETS')}/swe_nerc/swe_nerc-val.tsv",
            f"cat {env_variable('DIR_DATASETS')}/swe_nerc/raw_data/*-02.tsv "
            f"> {env_variable('DIR_DATASETS')}/swe_nerc/swe_nerc-test.tsv",
            f"cat {env_variable('DIR_DATASETS')}/swe_nerc/raw_data/*-0[!12].tsv "
            f"> {env_variable('DIR_DATASETS')}/swe_nerc/swe_nerc-train.tsv",
            f"cat {env_variable('DIR_DATASETS')}/swe_nerc/raw_data/*-[!0]?.tsv "
            f">> {env_variable('DIR_DATASETS')}/swe_nerc/swe_nerc-train.tsv",
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
        return dict()

    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        for phase in ["train", "val", "test"]:
            rows = self._read_original_file(phase)
            rows_iob2 = self._convert_iob1_to_iob2(rows)
            self._write_formatted_csv(phase, rows_iob2)

    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float], e.g. 0.3
        :return: -
        """
        # train -> train
        df_train = self._read_formatted_csvs(["train"])
        self._write_final_csv("train", df_train)

        # val  -> val
        df_val = self._read_formatted_csvs(["val"])
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
        file_name = {
            phase: f"swe_nerc-{phase}.tsv"
            for phase in ["train", "val", "test"]
        }
        file_path_original = join(self.dataset_path, file_name[phase])

        _rows = list()
        if os.path.isfile(file_path_original):
            with open(file_path_original) as f:
                for i, row in enumerate(f.readlines()):
                    _rows.append(row.split("\t"))
            print(f"\n> read {file_path_original}")
        else:
            raise Exception(f"> original file {file_path_original} could not be found.")

        _rows = [
            [
                "".join(row[0].split()),  # this replaces unwanted nbsp characters
                self.transform_tags(row)
            ]
            if len(row[0]) > 0 and len(row) > 1
            else list()
            for row in _rows
        ]

        return _rows

    @staticmethod
    def transform_tags(_row):
        assert len(_row) in [3, 4], f"ERROR! encountered row = {_row} that cannot be parsed."
        plain_tag = _row[1]
        if plain_tag == "O":
            return plain_tag
        else:
            if len(_row) == 3:
                return f"I-{plain_tag}"
            elif _row[3] == "B":
                return f"B-{plain_tag}"
            else:
                raise Exception(f"ERROR! encountered row = {_row} that cannot be parsed.")
