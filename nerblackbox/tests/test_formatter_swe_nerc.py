import pytest
from nerblackbox.modules.datasets.formatter.base_formatter import SENTENCES_ROWS
from nerblackbox.modules.datasets.formatter.swe_nerc_formatter import SweNercFormatter
from pkg_resources import resource_filename
from typing import List
import pandas as pd

import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestSweNercFormatter:

    formatter = SweNercFormatter()

    @pytest.mark.parametrize(
        "sentences_rows",
        [
            (
                [
                    [
                        ["Det", "I-PER"],
                        ["här", "O"],
                        ["är", "O"],
                        ["ett", "I-LOC"],
                        ["test", "O"],
                    ],
                    [["Med", "O"], ["två", "O"], ["meningar", "I-LOC"], [".", "O"]],
                    [["TestX", "I-PER"]],
                ]
            ),
        ],
    )
    def test_read_original_file(self, sentences_rows: SENTENCES_ROWS):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/original_data"
        )
        test_sentences_rows = self.formatter._read_original_file("test")
        assert (
            test_sentences_rows == sentences_rows
        ), f"ERROR! test_sentences_rows = {test_sentences_rows} != {sentences_rows}"

    @pytest.mark.parametrize(
        "sentences_rows_iob2",
        [
            (
                [
                    [["TestX", "B-PER"]],
                    [["Med", "O"], ["två", "O"], ["meningar", "B-LOC"], [".", "O"]],
                    [
                        ["Det", "B-PER"],
                        ["här", "O"],
                        ["är", "O"],
                        ["ett", "B-LOC"],
                        ["test", "O"],
                    ],
                ]
            ),
        ],
    )
    def test_format_data(self, sentences_rows_iob2: SENTENCES_ROWS):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/original_data"
        )
        test_sentences_rows_iob2 = self.formatter.format_data(
            shuffle=True, write_csv=False
        )
        assert (
            test_sentences_rows_iob2 == sentences_rows_iob2
        ), f"ERROR! test_sentences_rows_iob2 = {test_sentences_rows_iob2} != {sentences_rows_iob2}"

    @pytest.mark.parametrize(
        "row_list, tag, error",
        [
            (
                ["test", "O", "XYZ"],
                "O",
                False,
            ),
            (
                ["test-2", "PER", "XYZ"],
                "I-PER",
                False,
            ),
            (
                ["test-3", "PER", "XYZ", "B"],
                "B-PER",
                False,
            ),
            (
                ["test-3", "PER", "XYZ", "BXYZ"],
                "B-PER",
                True,
            ),
        ],
    )
    def test_transform_tags(self, row_list: List[str], tag: str, error: bool):
        if error:
            with pytest.raises(Exception):
                _ = self.formatter.transform_tags(row_list)
        else:
            test_tag = self.formatter.transform_tags(row_list)
            assert test_tag == tag, f"ERROR! test_tag = {test_tag} != {tag}"

    @pytest.mark.parametrize(
        "df_train, df_val, df_test",
        [
            (
                [
                    pd.DataFrame(
                        data=[
                            ["O O", "Mening 1"],
                            ["PER O", "Mening 2"],
                            ["O PER", "Mening 3"],
                            ["O PER", "Mening 4"],
                        ]
                    ),
                    pd.DataFrame(data=[["O O", "Mening 5"], ["PER O", "Mening 6"]]),
                    pd.DataFrame(data=[["O O", "Mening 7"], ["PER O", "Mening 8"]]),
                ]
            ),
        ],
    )
    def test_resplit_data(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
    ):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/formatted_data"
        )
        test_dfs = self.formatter.resplit_data(write_csv=False)
        assert (
            test_dfs is not None
        ), f"ERROR! resplit_data() returned None unexpectedly."
        test_df_train, test_df_val, test_df_test = test_dfs
        pd.testing.assert_frame_equal(
            test_df_train, df_train
        ), f"ERROR! test_resplit_data did not pass test for phase = train"
        pd.testing.assert_frame_equal(
            test_df_val, df_val
        ), f"ERROR! test_resplit_data did not pass test for phase = val"
        pd.testing.assert_frame_equal(
            test_df_test, df_test
        ), f"ERROR! test_resplit_data did not pass test for phase = test"
