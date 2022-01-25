import subprocess
from typing import List, Dict, Optional, Tuple
import pandas as pd

from nerblackbox.modules.datasets.formatter.base_formatter import (
    BaseFormatter,
    SENTENCES_ROWS,
)
from nerblackbox.modules.utils.env_variable import env_variable


class SweNercFormatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "swe_nerc"
        ner_tag_list = ["PRS", "LOC", "GRO", "EVN", "TME", "WRK", "SMP", "MNT"]
        super().__init__(ner_dataset, ner_tag_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def get_data(self, verbose: bool) -> None:  # pragma: no cover
        """
        I: get data

        Args:
            verbose: [bool]
        """
        bash_cmds = [
            f"mkdir {env_variable('DIR_DATASETS')}/_swe_nerc",
            f"curl -o {env_variable('DIR_DATASETS')}/_swe_nerc/swe_nerc.tar.gz "
            "https://spraakbanken.gu.se/lb/resurser/swe-nerc/Swe-NERC-v1.0.tar.gz",
            f"cd {env_variable('DIR_DATASETS')}/_swe_nerc && tar -xzf swe_nerc.tar.gz",
            f"mkdir {env_variable('DIR_DATASETS')}/swe_nerc/raw_data",
            f"mv {env_variable('DIR_DATASETS')}/_swe_nerc/Swe-NERC-v1.0/manually-tagged-part/*.tsv "
            f"{env_variable('DIR_DATASETS')}/swe_nerc/raw_data",
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

    def create_ner_tag_mapping(self) -> Dict[str, str]:
        """
        II: customize ner_training tag mapping if wanted

        Returns:
            ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return dict()

    def format_data(
        self, shuffle: bool = True, write_csv: bool = True
    ) -> Optional[SENTENCES_ROWS]:
        """
        III: format data

        Args:
            shuffle: whether to shuffle rows of dataset
            write_csv: whether to write dataset to csv (should always be True except for testing)

        Returns:
            sentences_rows_iob2: only if write_csv = False
        """
        for phase in ["train", "val", "test"]:
            sentences_rows_iob1 = self._read_original_file(phase)
            if shuffle:
                sentences_rows_iob1 = self._shuffle_dataset(phase, sentences_rows_iob1)

            sentences_rows_iob2 = self._convert_iob1_to_iob2(sentences_rows_iob1)
            if write_csv:  # pragma: no cover
                self._write_formatted_csv(phase, sentences_rows_iob2)
            else:
                return sentences_rows_iob2
        return None

    def set_original_file_paths(self) -> None:
        """
        III: format data

        Changed Attributes:
            file_paths: [Dict[str, str]], e.g. {'train': <path_to_train_csv>, 'val': ..}

        Returns: -
        """
        self.file_name = {
            phase: f"swe_nerc-{phase}.tsv" for phase in ["train", "val", "test"]
        }

    def _parse_row(self, _row: str) -> List[str]:
        """
        III: format data

        Args:
            _row: e.g. "Det PER X B"

        Returns:
            _row_list: e.g. ["Det", "PER", "X", "B"]
        """
        _row_list_raw = _row.replace("\t", " ").strip().split(" ")
        if (len(_row_list_raw) == 4 and _row_list_raw[-1] == "B") or (
            len(_row_list_raw) == 3
        ):
            return _row_list_raw
        elif len(_row_list_raw) > 2:
            return [" ".join(_row_list_raw[:-2]), _row_list_raw[-2], _row_list_raw[-1]]
        elif _row in ["\n", "\t\t\n"] or len(_row) == 0:
            return []
        elif _row.startswith("\t"):
            print(f"ATTENTION! row = {repr(_row)} observed -> will be skipped")
            return ["SKIP-THIS-TOKEN", "0", "PAD"]
        else:  # pragma: no cover
            raise Exception(f"ERROR! could not parse row = {repr(_row)}")

    def _format_original_file(self, _row_list: List[str]) -> Optional[List[str]]:
        """
        III: format data

        Args:
            _row_list: e.g. ["test", "PER", "X", "B"]

        Returns:
            _row_list_formatted: e.g. ["test", "B-PER"]
        """
        if not len(_row_list) in [3, 4]:  # pragma: no cover
            raise Exception(
                f"ERROR! row_list = {_row_list} should consist of 3 or 4 parts!"
            )

        if _row_list[0] != "SKIP-THIS-TOKEN":
            # transformation on _row_list[0] replaces unwanted nbsp characters
            _row_list_formatted = [
                "".join(_row_list[0].split()),
                self.transform_tags(_row_list),
            ]
            return _row_list_formatted
        else:
            return None

    def resplit_data(
        self, val_fraction: float = 0.0, write_csv: bool = True
    ) -> Optional[Tuple[pd.DataFrame, ...]]:
        """
        IV: resplit data

        Args:
            val_fraction: [float], e.g. 0.3
            write_csv: whether to write dataset to csv (should always be True except for testing)

        Returns:
            df_train: only if write_csv = False
            df_val:   only if write_csv = False
            df_test:  only if write_csv = False
        """
        # train -> train
        df_train = self._read_formatted_csvs(["train"])

        # val  -> val
        df_val = self._read_formatted_csvs(["val"])

        # test  -> test
        df_test = self._read_formatted_csvs(["test"])

        if write_csv:  # pragma: no cover
            self._write_final_csv("train", df_train)
            self._write_final_csv("val", df_val)
            self._write_final_csv("test", df_test)
            return None
        else:
            return df_train, df_val, df_test

    ####################################################################################################################
    # HELPER: READ ORIGINAL
    ####################################################################################################################
    @staticmethod
    def transform_tags(_row_list: List[str]) -> str:
        """
        Args:
            _row_list: ["test", "O", "XYZ"], ["test-2", "PER", "XYZ"] or ["test-3", "PER", "XYZ", "B"]

        Return:
            tag: e.g. "O", "I-PER" or "B-PER"
        """
        assert len(_row_list) in [
            3,
            4,
        ], f"ERROR! encountered row_list = {_row_list} that cannot be parsed."
        plain_tag = _row_list[1]
        if plain_tag == "O":
            return plain_tag
        else:
            if len(_row_list) == 3:
                return f"I-{plain_tag}"
            elif _row_list[3] == "B":
                return f"B-{plain_tag}"
            else:
                raise Exception(
                    f"ERROR! encountered row_list = {_row_list} that cannot be parsed."
                )
