import pytest
from typing import Dict

import pandas as pd
from nerblackbox.modules.datasets.formatter.swedish_ner_corpus_formatter import (
    SwedishNerCorpusFormatter,
)

from pkg_resources import resource_filename
import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestAnalyzer:

    analyzer = SwedishNerCorpusFormatter().analyzer
    analyzer.dataset_path = resource_filename(
        "nerblackbox", f"tests/test_data/final_data"
    )

    index = ["O", "PER", "ORG", "LOC", "MISC", "PRG", "ORG*"]
    columns = ["tags", "tags/sentence", "tags relative w/ 0", "tags relative w/o 0"]

    @pytest.mark.parametrize(
        "phase, num_sentences, stats_aggregated",
        [
            (
                "train",
                4,
                pd.DataFrame(data=[5, 3, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
            ),
            (
                "val",
                2,
                pd.DataFrame(data=[3, 1, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
            ),
            (
                "test",
                2,
                pd.DataFrame(data=[3, 1, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
            ),
        ],
    )
    def test_read_final_csv(
        self, phase: str, num_sentences: int, stats_aggregated: pd.DataFrame
    ):
        test_num_sentences, test_stats_aggregated = self.analyzer._read_final_csv(phase)
        assert (
            test_num_sentences == num_sentences
        ), f"ERROR! test_num_sentences = {test_num_sentences} not equal to num_sentences = {num_sentences}"
        pd.testing.assert_frame_equal(
            test_stats_aggregated, stats_aggregated
        ), f"ERROR! test_read_final_csv did not pass test for phase = {phase}"

    @pytest.mark.parametrize(
        "stats_aggregated_phase, num_tokens",
        [
            (
                pd.DataFrame(data=[3, 1, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
                4,
            ),
            (
                pd.DataFrame(data=[5, 3, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
                8,
            ),
        ],
    )
    def test_get_tokens(self, stats_aggregated_phase: pd.DataFrame, num_tokens):
        test_num_tokens = self.analyzer._get_num_tokens(stats_aggregated_phase)
        assert (
            test_num_tokens == num_tokens
        ), f"ERROR! test_num_tokens = {test_num_tokens} not equal to num_tokens = {num_tokens}"

    @pytest.mark.parametrize(
        "phase, stats_aggregated_phase, num_sentences, stats_aggregated_phase_extended",
        [
            (
                "train",
                pd.DataFrame(data=[5, 3, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
                4,
                pd.DataFrame(
                    data={
                        "tags": [5, 3, 0, 0, 0, 0, 0],
                        "tags/sentence": [
                            "{:.2f}".format(x)
                            for x in [5 / 4.0, 3 / 4.0, 0, 0, 0, 0, 0]
                        ],
                        "tags relative w/ 0": [
                            "{:.2f}".format(x)
                            for x in [5 / 8.0, 3 / 8.0, 0, 0, 0, 0, 0]
                        ],
                        "tags relative w/o 0": [
                            "{:.2f}".format(x) for x in [0, 3 / 3.0, 0, 0, 0, 0, 0]
                        ],
                    },
                    index=index,
                    columns=columns,
                ),
            ),
            (
                "val",
                pd.DataFrame(data=[3, 1, 0, 0, 0, 0, 0], index=index, columns=["tags"]),
                2,
                pd.DataFrame(
                    data={
                        "tags": [3, 1, 0, 0, 0, 0, 0],
                        "tags/sentence": [
                            "{:.2f}".format(x)
                            for x in [3 / 2.0, 1 / 2.0, 0, 0, 0, 0, 0]
                        ],
                        "tags relative w/ 0": [
                            "{:.2f}".format(x)
                            for x in [3 / 4.0, 1 / 4.0, 0, 0, 0, 0, 0]
                        ],
                        "tags relative w/o 0": [
                            "{:.2f}".format(x) for x in [0, 1 / 1.0, 0, 0, 0, 0, 0]
                        ],
                    },
                    index=index,
                    columns=columns,
                ),
            ),
        ],
    )
    def test_stats_aggregated_extend(
        self,
        phase: str,
        stats_aggregated_phase: pd.DataFrame,
        num_sentences: int,
        stats_aggregated_phase_extended: pd.DataFrame,
    ):
        test_stats_aggregated_phase_extended = self.analyzer._stats_aggregated_extend(
            stats_aggregated_phase, num_sentences
        )
        pd.testing.assert_frame_equal(
            test_stats_aggregated_phase_extended, stats_aggregated_phase_extended
        ), f"ERROR! test_stats_aggregated_extend pass test for phase = {phase}"

    @pytest.mark.parametrize(
        "stats_aggregated, num_tokens, num_sentences",
        [
            (
                {
                    "train": pd.DataFrame(
                        data={
                            "tags": [5, 3, 0, 0, 0, 0, 0],
                            "tags/sentence": [
                                "{:.2f}".format(x)
                                for x in [5 / 4.0, 3 / 4.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/ 0": [
                                "{:.2f}".format(x)
                                for x in [5 / 8.0, 3 / 8.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/o 0": [
                                "{:.2f}".format(x) for x in [0, 3 / 3.0, 0, 0, 0, 0, 0]
                            ],
                        },
                        index=index,
                        columns=columns,
                    ),
                    "val": pd.DataFrame(
                        data={
                            "tags": [3, 1, 0, 0, 0, 0, 0],
                            "tags/sentence": [
                                "{:.2f}".format(x)
                                for x in [3 / 2.0, 1 / 2.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/ 0": [
                                "{:.2f}".format(x)
                                for x in [3 / 4.0, 1 / 4.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/o 0": [
                                "{:.2f}".format(x) for x in [0, 1 / 1.0, 0, 0, 0, 0, 0]
                            ],
                        },
                        index=index,
                        columns=columns,
                    ),
                    "test": pd.DataFrame(
                        data={
                            "tags": [3, 1, 0, 0, 0, 0, 0],
                            "tags/sentence": [
                                "{:.2f}".format(x)
                                for x in [3 / 2.0, 1 / 2.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/ 0": [
                                "{:.2f}".format(x)
                                for x in [3 / 4.0, 1 / 4.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/o 0": [
                                "{:.2f}".format(x) for x in [0, 1 / 1.0, 0, 0, 0, 0, 0]
                            ],
                        },
                        index=index,
                        columns=columns,
                    ),
                    "total": pd.DataFrame(
                        data={
                            "tags": [11, 5, 0, 0, 0, 0, 0],
                            "tags/sentence": [
                                "{:.2f}".format(x)
                                for x in [11 / 8.0, 5 / 8.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/ 0": [
                                "{:.2f}".format(x)
                                for x in [11 / 16.0, 5 / 16.0, 0, 0, 0, 0, 0]
                            ],
                            "tags relative w/o 0": [
                                "{:.2f}".format(x)
                                for x in [0, 11 / 11.0, 0, 0, 0, 0, 0]
                            ],
                        },
                        index=index,
                        columns=columns,
                    ),
                },
                {"train": 8, "val": 4, "test": 4, "total": 16},
                {"train": 4, "val": 2, "test": 2, "total": 8},
            ),
        ],
    )
    def test_analyze_data(
        self,
        stats_aggregated: Dict[str, pd.DataFrame],
        num_tokens: Dict[str, int],
        num_sentences: Dict[str, int],
    ):
        self.analyzer.analyze_data(
            write_log=False
        )  # created attr: stats_aggregated, num_tokens, num_sentences
        for phase in ["train", "val", "test", "total"]:

            assert self.analyzer.num_tokens[phase] == num_tokens[phase], (
                f"ERROR! test_num_tokens = {self.analyzer.num_tokens[phase]} not equal to "
                f"num_tokens = {num_tokens[phase]} for phase = {phase}"
            )

            assert self.analyzer.num_sentences[phase] == num_sentences[phase], (
                f"ERROR! test_num_sentences = {self.analyzer.num_sentences[phase]} not equal to "
                f"num_sentences = {num_sentences[phase]} for phase = {phase}"
            )

            pd.testing.assert_frame_equal(
                self.analyzer.stats_aggregated[phase], stats_aggregated[phase]
            ), f"ERROR! test_analyze_data / stats_aggregated did not pass test for phase = {phase}"
