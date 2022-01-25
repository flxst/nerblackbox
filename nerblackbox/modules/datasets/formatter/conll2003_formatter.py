import requests
from os.path import join, isfile
from typing import Dict, List, Optional, Tuple
import pandas as pd

from nerblackbox.modules.datasets.formatter.base_formatter import (
    BaseFormatter,
    SENTENCES_ROWS,
)


class CoNLL2003Formatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "conll2003"
        ner_tag_list = ["PER", "ORG", "LOC", "MISC"]
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
        url_base = "https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/"
        targets = ["eng.train", "eng.testa", "eng.testb"]

        for target in targets:
            target_file = join(self.dataset_path, target)

            # fetch tgz from url
            if isfile(target_file):
                if verbose:
                    print(f".. file at {target_file} already exists")
            else:
                url = url_base + target
                myfile = requests.get(url, allow_redirects=True)
                open(target_file, "wb").write(myfile.content)
                if verbose:
                    print(f".. file fetched from {url} and saved at {target_file}")

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
            "train": "eng.train",
            "val": "eng.testa",
            "test": "eng.testb",
        }

    def _parse_row(self, _row: str) -> List[str]:
        """
        III: format data

        Args:
            _row: e.g. "Det PER X B"

        Returns:
            _row_list: e.g. ["Det", "PER", "X", "B"]
        """
        return _row.strip().split()

    def _format_original_file(self, _row_list: List[str]) -> Optional[List[str]]:
        """
        III: format data

        Args:
            _row_list: e.g. ["test", "PER", "X", "B"]

        Returns:
            _row_list_formatted: e.g. ["test", "B-PER"]
        """
        if not len(_row_list) in [4]:  # pragma: no cover
            raise Exception(f"ERROR! row_list = {_row_list} should consist of 4 parts!")

        if _row_list[0] != "-DOCSTART-":
            _row_list_formatted = [_row_list[0], _row_list[-1]]
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
            self._write_final_csv("val", df_val)
            self._write_final_csv("train", df_train)
            self._write_final_csv("test", df_test)
            return None
        else:
            return df_train, df_val, df_test
