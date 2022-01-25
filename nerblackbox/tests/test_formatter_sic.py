import pytest
from nerblackbox.modules.datasets.formatter.base_formatter import SENTENCES_ROWS
from nerblackbox.modules.datasets.formatter.sic_formatter import SICFormatter
from pkg_resources import resource_filename
import pandas as pd

import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestSicFormatter:

    formatter = SICFormatter()

    @pytest.mark.parametrize(
        "sentences_rows",
        [
            (
                [
                    [
                        ["Det", "B-PER"],
                        ["här", "O"],
                        ["är", "O"],
                        ["ett", "B-LOC"],
                        ["test", "O"],
                    ],
                    [["Med", "O"], ["två", "O"], ["meningar", "B-LOC"], [".", "O"]],
                ]
            ),
        ],
    )
    def test_read_original_file(self, sentences_rows: SENTENCES_ROWS):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/original_data"
        )
        test_sentences_rows = self.formatter._read_original_file("train")
        assert (
            test_sentences_rows == sentences_rows
        ), f"ERROR! test_sentences_rows = {test_sentences_rows} != {sentences_rows}"

    @pytest.mark.parametrize(
        "sentences_rows",
        [
            (
                [
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
    def test_format_data(self, sentences_rows: SENTENCES_ROWS):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/original_data"
        )
        test_sentences_rows = self.formatter.format_data(shuffle=True, write_csv=False)
        assert (
            test_sentences_rows == sentences_rows
        ), f"ERROR! test_sentences_rows = {test_sentences_rows} != {sentences_rows}"

    @pytest.mark.parametrize(
        "bio, tag, transformed_tag",
        [
            (
                "B",
                "PER",
                "B-PER",
            ),
            (
                "O",
                "_",
                "O",
            ),
        ],
    )
    def test_transform_tags(self, bio: str, tag: str, transformed_tag: str):
        test_transformed_tag = self.formatter.transform_tags(bio, tag)
        assert (
            test_transformed_tag == transformed_tag
        ), f"ERROR! test_transformed_tag = {test_transformed_tag} != {transformed_tag}"

    @pytest.mark.parametrize(
        "val_fraction, df_train, df_val, df_test",
        [
            (
                [
                    0.25,
                    pd.DataFrame(
                        data=[["O O", "Mening 1"], ["PER O", "Mening 2"]],
                        index=pd.RangeIndex(0, 2),
                    ),
                    pd.DataFrame(
                        data=[["O PER", "Mening 3"]], index=pd.RangeIndex(2, 3)
                    ),
                    pd.DataFrame(
                        data=[["O PER", "Mening 4"]], index=pd.RangeIndex(3, 4)
                    ),
                ]
            ),
        ],
    )
    def test_resplit_data(
        self,
        val_fraction: float,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        self.formatter.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/formatted_data"
        )
        test_dfs = self.formatter.resplit_data(
            val_fraction=val_fraction, write_csv=False
        )
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
