import pytest
from nerblackbox.modules.datasets.formatter.base_formatter import SENTENCES_ROWS
from nerblackbox.modules.datasets.formatter.sucx_formatter import SUCXFormatter
from pkg_resources import resource_filename
import pandas as pd

import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestSucxFormatter:

    formatter = SUCXFormatter("original_cased")

    @pytest.mark.parametrize(
        "sentences_rows",
        [
            (
                [
                    [
                        ["I", "O"],
                        ["850-modellen", "O"],
                        ["har", "O"],
                        ["man", "O"],
                        ["valt", "O"],
                        ["en", "O"],
                        ["tredje", "O"],
                        ["variant", "O"],
                        [",", "O"],
                        ["Delta-link", "B-product"],
                        [".", "O"],
                    ],
                    [
                        ["Audi", "B-product"],
                        ["Coupé", "I-product"],
                        ["Quattro", "I-product"],
                        ["20V", "B-product"],
                    ],
                ]
            ),
        ],
    )
    def test_format_data(self, sentences_rows: SENTENCES_ROWS):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/original_data"
        )
        test_sentences_rows = self.formatter.format_data(shuffle=True, write_csv=False)
        assert (
            test_sentences_rows == sentences_rows
        ), f"ERROR! test_sentences_rows = {test_sentences_rows} != {sentences_rows}"

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
