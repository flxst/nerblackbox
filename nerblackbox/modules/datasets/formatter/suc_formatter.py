from typing import List, Dict, Optional, Tuple
import pandas as pd

from nerblackbox.modules.datasets.formatter.base_formatter import (
    BaseFormatter,
    SENTENCES_ROWS,
)


class SUCFormatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "suc"
        ner_tag_list = [
            "person",
            "place",
            "inst",
            "work",
            "misc",
        ]  # misc: see create_ner_tag_mapping()
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
        print("(suc: nothing to do)")

    def create_ner_tag_mapping(self) -> Dict[str, str]:
        """
        II: customize ner_training tag mapping if wanted

        Returns:
            ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return dict()
        # return {
        #     'animal': 'misc',
        #     'myth': 'misc',
        #     'product': 'misc',
        #     'event': 'misc',
        #     'other': 'misc',
        # }

    def format_data(
        self, shuffle: bool = True, write_csv: bool = True
    ) -> Optional[SENTENCES_ROWS]:
        """
        III: format data

        Args:
            shuffle: whether to shuffle rows of dataset
            write_csv: whether to write dataset to csv (should always be True except for testing)

        Returns:
            sentences_rows: only if write_csv = False
        """
        for phase in ["train", "val", "test"]:
            sentences_rows = self._read_original_file(phase)
            if shuffle:
                sentences_rows = self._shuffle_dataset(phase, sentences_rows)

            if write_csv:  # pragma: no cover
                self._write_formatted_csv(phase, sentences_rows)
            else:
                return sentences_rows
        return None

    def set_original_file_paths(self) -> None:
        """
        III: format data

        Changed Attributes:
            file_paths: [Dict[str, str]], e.g. {'train': <path_to_train_csv>, 'val': ..}

        Returns: -
        """
        self.file_name = {
            "train": "suc-train.conll",
            "val": "suc-dev.conll",
            "test": "suc-test.conll",
        }

    def _parse_row(self, _row: str) -> List[str]:
        """
        III: format data

        Args:
            _row: e.g. "Det PER X B"

        Returns:
            _row_list: e.g. ["Det", "PER", "X", "B"]
        """
        _row_list = _row.split("\t")
        if _row_list == ["\n"]:
            return []
        else:
            return _row_list

    def _format_original_file(self, _row_list: List[str]) -> Optional[List[str]]:
        """
        III: format data

        Args:
            _row_list: e.g. ["test", "PER", "X", "B"]

        Returns:
            _row_list_formatted: e.g. ["test", "B-PER"]
        """
        if not len(_row_list) in [13]:
            print(
                f"ATTENTION! row_list = {_row_list} should consist of 13 parts! -> treat as empty row"
            )
            return None

        _row_list_formatted = [
            _row_list[1],
            self.transform_tags(_row_list[-3], _row_list[-2]),
        ]

        return _row_list_formatted

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
    def transform_tags(bio: str, tag: str) -> str:
        """
        Args:
            bio: e.g. 'O', 'B', 'I'
            tag: e.g. '_', 'person', ..

        Returns:
            transformed tag: e.g. 'O', 'B-person', ..
        """
        if bio == "O":
            return "O"
        else:
            return f"{bio}-{tag}"
