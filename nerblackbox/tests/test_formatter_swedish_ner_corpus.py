import pytest
from nerblackbox.modules.datasets.formatter.base_formatter import SENTENCES_ROWS
from nerblackbox.modules.datasets.formatter.swedish_ner_corpus_formatter import (
    SwedishNerCorpusFormatter,
)
from pkg_resources import resource_filename
import pandas as pd

import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestSwedishNerCorpusFormatter:

    formatter = SwedishNerCorpusFormatter()

    @pytest.mark.parametrize(
        "sentences_rows",
        [
            (
                [
                    [
                        ["Det", "PER"],
                        ["här", "0"],
                        ["är", "0"],
                        ["ett", "LOC"],
                        ["test", "0"],
                    ],
                    [["Med", "0"], ["två", "0"], ["meningar", "LOC"], [".", "0"]],
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
        "sentences_rows",
        [
            (
                [
                    [["Med", "0"], ["två", "0"], ["meningar", "LOC"], [".", "0"]],
                    [
                        ["Det", "PER"],
                        ["här", "0"],
                        ["är", "0"],
                        ["ett", "LOC"],
                        ["test", "0"],
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
        "val_fraction, df_train, df_val, df_test",
        [
            (
                [
                    0.5,
                    pd.DataFrame(
                        data=[["O O", "Mening 1"], ["PER O", "Mening 2"]],
                        index=pd.RangeIndex(0, 2),
                    ),
                    pd.DataFrame(
                        data=[["O PER", "Mening 3"], ["O PER", "Mening 4"]],
                        index=pd.RangeIndex(2, 4),
                    ),
                    pd.DataFrame(
                        data=[["O O", "Mening 7"], ["PER O", "Mening 8"]],
                        index=pd.RangeIndex(0, 2),
                    ),
                ]
            ),
            (
                [
                    0.25,
                    pd.DataFrame(
                        data=[
                            ["O O", "Mening 1"],
                            ["PER O", "Mening 2"],
                            ["O PER", "Mening 3"],
                        ],
                        index=pd.RangeIndex(0, 3),
                    ),
                    pd.DataFrame(
                        data=[["O PER", "Mening 4"]], index=pd.RangeIndex(3, 4)
                    ),
                    pd.DataFrame(
                        data=[["O O", "Mening 7"], ["PER O", "Mening 8"]],
                        index=pd.RangeIndex(0, 2),
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
