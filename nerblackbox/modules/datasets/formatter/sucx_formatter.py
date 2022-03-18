from typing import List, Dict, Optional, Tuple
import pandas as pd
from datasets import load_dataset
from os.path import join
from datasets import DatasetDict

from nerblackbox.modules.datasets.formatter.base_formatter import (
    BaseFormatter,
    SENTENCES_ROWS,
)

SUCX_SUBSETS = [
    "original_cased",
    "original_lower",
    "original_lower_mix",
    "simple_cased",
    "simple_lower",
    "simple_lower_mix",
]
SUCX_NER_TAG_LIST = {
    "original": [
        "person",
        "place",
        "inst",
        "work",
        "product",
        "animal",
        "event",
        "myth",
        "other",
    ],
    "simple": ["PER", "LOC", "ORG", "WRK", "OBJ", "PRS", "EVN", "PRS", "MSR", "TME"],
}


class SUCXFormatter(BaseFormatter):
    def __init__(self, ner_dataset_subset: str):
        ner_dataset = "sucx"
        assert (
            ner_dataset_subset in SUCX_SUBSETS
        ), f"ERROR! subset = {ner_dataset_subset} unknown."

        tag_group = ner_dataset_subset.split("_")[0]
        assert tag_group in [
            "original",
            "simple",
        ], f"ERROR! tag_group = {tag_group} should be original or simple."
        ner_tag_list = SUCX_NER_TAG_LIST[tag_group]

        super().__init__(ner_dataset, ner_tag_list, ner_dataset_subset)
        self.ner_dataset_subset = ner_dataset_subset

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def get_data(self, verbose: bool) -> None:  # pragma: no cover
        """
        I: get data

        Args:
            verbose: [bool]
        """
        dataset = load_dataset("KBLab/sucx3_ner", self.ner_dataset_subset)
        assert isinstance(
            dataset, DatasetDict
        ), f"ERROR! type(dataset) = {type(dataset)} expected to be DatasetDict."
        dataset["val"] = dataset["validation"]
        for phase in ["train", "val", "test"]:
            sentences_rows_formatted = []
            for sample in dataset[phase]:
                sentences_rows_formatted.append(
                    (" ".join(sample["ner_tags"]), " ".join(sample["tokens"]))
                )

            df = pd.DataFrame(sentences_rows_formatted)
            assert (
                len(df) == dataset[phase].num_rows
            ), f"ERROR! number of rows = {len(df)} != {dataset[phase].num_rows}"
            file_path = join(self.dataset_path, f"{phase}_original.csv")
            df.to_csv(file_path, sep="\t", header=False, index=False)

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
            sentences_rows: only if write_csv = False
        """
        self.set_original_file_paths()
        for phase in ["train", "val", "test"]:
            file_path_original = join(self.dataset_path, self.file_name[phase])
            df = pd.read_csv(file_path_original, sep="\t", header=None)
            _sentences_rows = [_np_array for _np_array in df.values]
            sentences_rows = [
                [[word, tag] for word, tag in zip(words.split(), tags.split())]
                for tags, words in _sentences_rows
            ]
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
            "train": "train_original.csv",
            "val": "val_original.csv",
            "test": "test_original.csv",
        }

    def _parse_row(self, _row: str) -> List[str]:
        """
        III: format data

        Args:
            _row: e.g. "Det PER X B"

        Returns:
            _row_list: e.g. ["Det", "PER", "X", "B"]
        """
        pass

    def _format_original_file(self, _row_list: List[str]) -> Optional[List[str]]:
        """
        III: format data

        Args:
            _row_list: e.g. ["B-product I-product I-product B-product	Audi Coupé Quattro 20V"]

        Returns:
            _row_list_formatted: e.g. ["test", "B-PER"]
        """
        pass

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
