import pytest
from typing import List, Tuple

import pandas as pd
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter
from nerblackbox.modules.datasets.formatter.base_formatter import (
    BaseFormatter,
    SENTENCES_ROWS,
    SENTENCES_ROWS_PRETOKENIZED,
)
from nerblackbox.modules.datasets.formatter.conll2003_formatter import (
    CoNLL2003Formatter,
)
from nerblackbox.modules.datasets.formatter.swe_nerc_formatter import SweNercFormatter
from nerblackbox.modules.datasets.formatter.swedish_ner_corpus_formatter import (
    SwedishNerCorpusFormatter,
)
from nerblackbox.modules.datasets.formatter.sic_formatter import SICFormatter
from nerblackbox.modules.datasets.formatter.suc_formatter import SUCFormatter
from nerblackbox.modules.datasets.formatter.huggingface_datasets_formatter import (
    HuggingfaceDatasetsFormatter,
)

from pkg_resources import resource_filename
import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestAutoFormatter:
    @pytest.mark.parametrize(
        "ner_dataset, error",
        [
            ("swedish_ner_corpus", False),
            ("conll2003", False),
            ("sic", False),
            ("suc", False),
            ("swe_nerc", False),
            ("xyz", True),
            ("ehealth_kd", False),
            ("sent_comp", True),
        ],
    )
    def test_for_dataset(self, ner_dataset: str, error: bool):
        if error:
            with pytest.raises(Exception):
                _ = AutoFormatter.for_dataset(ner_dataset=ner_dataset)
        else:
            auto_formatter = AutoFormatter.for_dataset(ner_dataset=ner_dataset)
            assert isinstance(
                auto_formatter, BaseFormatter
            ), f"ERROR! type(auto_formatter) = {type(auto_formatter)} != BaseFormatter"
            assert (
                auto_formatter.ner_dataset == ner_dataset
            ), f"ERROR! auto_formatter.ner_dataset = {auto_formatter.ner_dataset} != {ner_dataset}"


class TestBaseFormatter:

    base_formatter = SwedishNerCorpusFormatter()
    base_formatter.dataset_path = resource_filename(
        "nerblackbox", f"tests/test_data/formatted_data"
    )

    @pytest.mark.parametrize(
        "phases, df_formatted",
        [
            (
                ["train"],
                pd.DataFrame(
                    data=[
                        ["O O", "Mening 1"],
                        ["PER O", "Mening 2"],
                        ["O PER", "Mening 3"],
                        ["O PER", "Mening 4"],
                    ]
                ),
            ),
            (["val"], pd.DataFrame(data=[["O O", "Mening 5"], ["PER O", "Mening 6"]])),
            (["test"], pd.DataFrame(data=[["O O", "Mening 7"], ["PER O", "Mening 8"]])),
            (
                ["val", "test"],
                pd.DataFrame(
                    data=[
                        ["O O", "Mening 5"],
                        ["PER O", "Mening 6"],
                        ["O O", "Mening 7"],
                        ["PER O", "Mening 8"],
                    ]
                ),
            ),
            (
                ["test", "val"],
                pd.DataFrame(
                    data=[
                        ["O O", "Mening 7"],
                        ["PER O", "Mening 8"],
                        ["O O", "Mening 5"],
                        ["PER O", "Mening 6"],
                    ]
                ),
            ),
        ],
    )
    def test_read_formatted_csvs(self, phases: List[str], df_formatted: pd.DataFrame):
        test_df_formatted = self.base_formatter._read_formatted_csvs(phases)
        pd.testing.assert_frame_equal(
            test_df_formatted, df_formatted
        ), f"ERROR! test_read_formatted did not pass test for phases = {phases}"

    @pytest.mark.parametrize(
        "df_original, val_fraction, df_new, df_val",
        [
            (
                pd.DataFrame(data=[[1, 2], [3, 4]], columns=["A", "B"], index=[0, 1]),
                0.5,
                pd.DataFrame(data=[[1, 2]], columns=["A", "B"], index=[0]),
                pd.DataFrame(data=[[3, 4]], columns=["A", "B"], index=[1]),
            ),
            (
                pd.DataFrame(
                    data=[[1, 2], [3, 4], [5, 6], [7, 8]],
                    columns=["A", "B"],
                    index=["a", "b", "c", "d"],
                ),
                0.25,
                pd.DataFrame(
                    data=[[1, 2], [3, 4], [5, 6]],
                    columns=["A", "B"],
                    index=["a", "b", "c"],
                ),
                pd.DataFrame(data=[[7, 8]], columns=["A", "B"], index=["d"]),
            ),
        ],
    )
    def test_split_off_validation_set(
        self,
        df_original: pd.DataFrame,
        val_fraction: float,
        df_new: pd.DataFrame,
        df_val: pd.DataFrame,
    ):
        test_df_new, test_df_val = self.base_formatter._split_off_validation_set(
            df_original, val_fraction
        )
        pd.testing.assert_frame_equal(
            test_df_new, df_new
        ), f"ERROR! test_split_off_validation_set / df_new did not pass test"
        pd.testing.assert_frame_equal(
            test_df_val, df_val
        ), f"ERROR! test_split_off_validation_set / df_val did not pass test"

    @pytest.mark.parametrize(
        "sentences_rows, sentences_rows_formatted",
        [
            (
                [
                    [["Inger", "B-PER"], ["Nilsson", "I-PER"], ["Frida", "B-PER"]],
                    [["är", "O"], ["Inger", "B-PER"], ["och", "O"]],
                ],
                [
                    ("B-PER I-PER B-PER", "Inger Nilsson Frida"),
                    ("O B-PER O", "är Inger och"),
                ],
            )
        ],
    )
    def test_format_sentences_rows(
        self,
        sentences_rows: SENTENCES_ROWS_PRETOKENIZED,
        sentences_rows_formatted: List[Tuple[str, str]],
    ):
        test_sentences_rows_formatted = self.base_formatter._format_sentences_rows(
            sentences_rows
        )
        assert (
            test_sentences_rows_formatted == sentences_rows_formatted
        ), f"ERROR! test_sentences_rows_formatted = {test_sentences_rows_formatted} != {sentences_rows_formatted}"

    @pytest.mark.parametrize(
        "sentences_rows_iob1, sentences_rows_iob2",
        [
            (
                [[["Inger", "I-PER"], ["säger", "0"]]],
                [[["Inger", "B-PER"], ["säger", "0"]]],
            ),
            (
                [[["Inger", "I-PER"], ["Nilsson", "I-PER"], ["Frida", "B-PER"]]],
                [[["Inger", "B-PER"], ["Nilsson", "I-PER"], ["Frida", "B-PER"]]],
            ),
            (
                [[["är", "O"], ["Inger", "I-PER"], ["och", "O"]]],
                [[["är", "O"], ["Inger", "B-PER"], ["och", "O"]]],
            ),
        ],
    )
    def test_convert_iob1_to_iob2(
        self,
        sentences_rows_iob1: SENTENCES_ROWS_PRETOKENIZED,
        sentences_rows_iob2: SENTENCES_ROWS_PRETOKENIZED,
    ):
        test_sentences_rows_iob2 = self.base_formatter._convert_iob1_to_iob2(
            sentences_rows_iob1
        )
        assert (
            test_sentences_rows_iob2 == sentences_rows_iob2
        ), f"ERROR! test_sentences_rows_iob2 = {test_sentences_rows_iob2} != {sentences_rows_iob2}"

    @pytest.mark.parametrize(
        "phase, sentences_rows, sentences_rows_shuffled",
        [
            (
                "train",
                [
                    [["Inger", "B-PER"], ["säger", "0"]],
                    [["Inger", "B-PER"], ["Nilsson", "I-PER"], ["Frida", "B-PER"]],
                    [["är", "O"], ["Inger", "B-PER"], ["och", "O"]],
                ],
                [
                    [["är", "O"], ["Inger", "B-PER"], ["och", "O"]],
                    [["Inger", "B-PER"], ["Nilsson", "I-PER"], ["Frida", "B-PER"]],
                    [["Inger", "B-PER"], ["säger", "0"]],
                ],
            ),
        ],
    )
    def test_shuffle_dataset(
        self,
        phase: str,
        sentences_rows: SENTENCES_ROWS_PRETOKENIZED,
        sentences_rows_shuffled: SENTENCES_ROWS_PRETOKENIZED,
    ):
        test_shuffled_sentences_rows = self.base_formatter._shuffle_dataset(
            phase, sentences_rows
        )
        assert test_shuffled_sentences_rows == sentences_rows_shuffled, (
            f"ERROR! test_shuffled_sentences_rows = {test_shuffled_sentences_rows} "
            f"!= {sentences_rows_shuffled} = sentences_rows_shuffled"
        )


class TestAllFormatters:

    formatters = {
        "conll2003": CoNLL2003Formatter(),
        "swe_nerc": SweNercFormatter(),
        "swedish_ner_corpus": SwedishNerCorpusFormatter(),
        "sic": SICFormatter(),
        "suc": SUCFormatter(),
        "ehealth_kd": HuggingfaceDatasetsFormatter("ehealth_kd"),
    }

    @pytest.mark.parametrize(
        "ner_dataset",
        [
            "conll2003",
            "swe_nerc",
            "swedish_ner_corpus",
            "sic",
            "suc",
            "ehealth_kd",
        ],
    )
    def test_create_ner_tag_mapping(self, ner_dataset: str):
        ner_tag_mapping = self.formatters[ner_dataset].create_ner_tag_mapping()
        assert isinstance(
            ner_tag_mapping, dict
        ), f"ERROR! ner_tag_mapping = {ner_tag_mapping} for ner_dataset = {ner_dataset} is not a dict!"
