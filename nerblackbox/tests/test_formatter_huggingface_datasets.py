import pytest
import pandas as pd
from nerblackbox.modules.datasets.formatter.huggingface_datasets_formatter import (
    HuggingfaceDatasetsFormatter,
)
from nerblackbox.modules.datasets.formatter.base_formatter import SENTENCES_ROWS
from pkg_resources import resource_filename
from typing import List, Tuple, Optional, Dict

import os
from os.path import abspath, dirname, join

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR


class TestHuggingfaceDatasetsFormatter:

    formatter_pretokenized = HuggingfaceDatasetsFormatter(
        "conll2003"
    )  # use any ner_dataset that works
    formatter_pretokenized_subset = HuggingfaceDatasetsFormatter(
        "wikiann", "ace"
    )  # use any ner_dataset that works
    formatter_unpretokenized = HuggingfaceDatasetsFormatter(
        "ehealth_kd"
    )  # use any ner_dataset that works

    ####################################################################################################################
    # check_existence()
    ####################################################################################################################
    @pytest.mark.parametrize(
        "ner_dataset, ner_dataset_subset, existence",
        [
            (["ehealth_kd", "", True]),
            (["conll2003", "", True]),
            (["conll2003", "this_subset_does_not_exist", False]),
            (["wikiann", "ace", True]),
            (["wikiann", "this_subset_does_not_exist", False]),
            (["swedish_ner_corpus", "", True]),
            (["sent_comp", "", True]),
            (["this_dataset_does_not_exist", "", False]),
        ],
    )
    def test_check_existence(
        self, ner_dataset: str, ner_dataset_subset: str, existence: bool
    ):
        test_existence, _ = HuggingfaceDatasetsFormatter.check_existence(
            ner_dataset, ner_dataset_subset
        )
        assert (
            test_existence == existence
        ), f"ERROR! test_existence = {test_existence} != {existence} for ner_dataset = {ner_dataset}"

    ####################################################################################################################
    # check_compatibility()
    ####################################################################################################################
    @pytest.mark.parametrize(
        "ner_dataset, ner_dataset_subset, compatibility",
        [
            (["ehealth_kd", "", True]),
            (["conll2003", "", True]),
            (["wikiann", "ace", True]),
            (
                [
                    "swedish_ner_corpus",
                    "",
                    False,  # only train & test on huggingface datasets
                ]
            ),
            (
                ["sent_comp", "", False]
            ),  # only train & validation on huggingface datasets
        ],
    )
    def test_check_compatibility(
        self, ner_dataset: str, ner_dataset_subset: str, compatibility: bool
    ):
        test_compability, _ = HuggingfaceDatasetsFormatter.check_compatibility(
            ner_dataset, ner_dataset_subset
        )
        assert (
            test_compability == compatibility
        ), f"ERROR! test_compatibility = {test_compability} != {compatibility} for ner_dataset = {ner_dataset}"

    ####################################################################################################################
    # check_implementation()
    ####################################################################################################################
    @pytest.mark.parametrize(
        "ner_dataset, ner_dataset_subset, implementation",
        [
            (["ehealth_kd", "", True]),  # ['entities'][0]['ent_label'].names
            (["conll2003", "", True]),  # ['ner_tags'].feature.names
            (["wikiann", "ace", True]),  # ['ner_tags'].feature.names
            (["swedish_ner_corpus", "", True]),  # ['ner_tags'].feature.names
            (["sent_comp", "", False]),
        ],
    )
    def test_check_implementation(
        self, ner_dataset: str, ner_dataset_subset: str, implementation: bool
    ):
        test_implementation, _ = HuggingfaceDatasetsFormatter.check_implementation(
            ner_dataset, ner_dataset_subset
        )
        assert (
            test_implementation == implementation
        ), f"ERROR! test_implementation = {test_implementation} != {implementation} for ner_dataset = {ner_dataset}"

    ####################################################################################################################
    # get_info()
    ####################################################################################################################
    @pytest.mark.parametrize(
        "ner_dataset, ner_dataset_subset, info",
        [
            (
                [
                    "ehealth_kd",
                    "",
                    (
                        True,  # ['entities'][0]['ent_label'].names
                        ["Concept", "Action", "Predicate", "Reference"],
                        False,
                        {
                            "text": "sentence",
                            "tags": "entities",
                            "mapping": {
                                "ent_text": "token",
                                "ent_label": "tag",
                                "start_character": "char_start",
                                "end_character": "char_end",
                            },
                        },
                    ),
                ]
            ),
            (
                [
                    "conll2003",
                    "",
                    (
                        True,  # ['ner_tags'].feature.names
                        [
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                            "B-MISC",
                            "I-MISC",
                        ],
                        True,
                        {"text": "tokens", "tags": "ner_tags", "mapping": None},
                    ),
                ]
            ),
            (
                [
                    "wikiann",
                    "ace",
                    (
                        True,  # ['ner_tags'].feature.names
                        [
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                        ],
                        True,
                        {"text": "tokens", "tags": "ner_tags", "mapping": None},
                    ),
                ]
            ),
            (
                [
                    "swedish_ner_corpus",
                    "",
                    (
                        True,  # ['ner_tags'].feature.names
                        ["0", "LOC", "MISC", "ORG", "PER"],
                        True,
                        {"text": "tokens", "tags": "ner_tags", "mapping": None},
                    ),
                ]
            ),
            (
                [
                    "sent_comp",
                    "",
                    (
                        False,
                        None,
                        None,
                        None,
                    ),
                ]
            ),
        ],
    )
    def test_get_info(
        self,
        ner_dataset: str,
        ner_dataset_subset: str,
        info: Tuple[
            bool, Optional[List[str]], Optional[bool], Optional[Dict[str, str]]
        ],
    ):
        print(ner_dataset)
        test_info = HuggingfaceDatasetsFormatter.get_infos(
            ner_dataset, ner_dataset_subset
        )
        print(test_info)
        for k in range(len(test_info)):
            assert (
                test_info[k] == info[k]
            ), f"ERROR! test_info[{k}] = {test_info[k]} != {info[k]} for ner_dataset = {ner_dataset}"

    ####################################################################################################################
    # get_ner_tag_list()
    ####################################################################################################################
    @pytest.mark.parametrize(
        "ner_dataset, ner_dataset_subset, ner_tag_list",
        [
            (["conll2003", "", ["LOC", "MISC", "ORG", "PER"]]),
            (["wikiann", "ace", ["LOC", "ORG", "PER"]]),
            (["ehealth_kd", "", ["Action", "Concept", "Predicate", "Reference"]]),
        ],
    )
    def test_get_ner_tag_list(
        self, ner_dataset: str, ner_dataset_subset: str, ner_tag_list: List[str]
    ):
        formatter = HuggingfaceDatasetsFormatter(ner_dataset, ner_dataset_subset)
        assert (
            formatter.ner_tag_list == ner_tag_list
        ), f"ERROR! test_ner_tag_list = {formatter.ner_tag_list} != {ner_tag_list}"

    ####################################################################################################################
    # ---
    ####################################################################################################################
    @pytest.mark.parametrize(
        "pretokenized, sentences_rows_iob2",
        [
            (
                True,
                [
                    [["Peter", "B-PER"], ["Blackburn", "I-PER"]],
                    [
                        ["EU", "B-ORG"],
                        ["rejects", "O"],
                        ["German", "B-MISC"],
                        ["call", "O"],
                        ["to", "O"],
                        ["boycott", "O"],
                        ["British", "B-MISC"],
                        ["lamb", "O"],
                        [".", "O"],
                    ],
                ],
            ),
            (
                False,
                [
                    {
                        "text": "La mayoría de las arritmias son el resultado de problemas en el sistema eléctrico del corazón.",
                        "tags": [
                            {
                                "token": "mayoría",
                                "tag": "Predicate",
                                "char_start": 3,
                                "char_end": 10,
                            },
                            {
                                "token": "arritmias",
                                "tag": "Concept",
                                "char_start": 18,
                                "char_end": 27,
                            },
                            {
                                "token": "problemas",
                                "tag": "Action",
                                "char_start": 48,
                                "char_end": 57,
                            },
                            {
                                "token": "sistema eléctrico",
                                "tag": "Concept",
                                "char_start": 64,
                                "char_end": 81,
                            },
                            {
                                "token": "corazón",
                                "tag": "Concept",
                                "char_start": 86,
                                "char_end": 93,
                            },
                        ],
                    },
                    {
                        "text": "En la leucemia linfocítica crónica, hay demasiados linfocitos, un tipo de glóbulos blancos.",
                        "tags": [
                            {
                                "token": "leucemia linfocítica crónica",
                                "tag": "Concept",
                                "char_start": 6,
                                "char_end": 34,
                            },
                            {
                                "token": "linfocitos",
                                "tag": "Concept",
                                "char_start": 51,
                                "char_end": 61,
                            },
                            {
                                "token": "glóbulos blancos",
                                "tag": "Concept",
                                "char_start": 74,
                                "char_end": 90,
                            },
                        ],
                    },
                ],
            ),
        ],
    )
    def test_format_data(self, pretokenized: bool, sentences_rows_iob2: SENTENCES_ROWS):

        formatter = (
            self.formatter_pretokenized
            if pretokenized
            else self.formatter_unpretokenized
        )

        formatter.get_data(verbose=False)
        if pretokenized:
            formatter.sentences_rows_pretokenized = {
                phase: formatter.sentences_rows_pretokenized[phase][:2]
                for phase in formatter.PHASES
            }
        else:
            formatter.sentences_rows_unpretokenized = {
                phase: formatter.sentences_rows_unpretokenized[phase][:2]
                for phase in formatter.PHASES
            }

        test_sentences_rows_iob2 = formatter.format_data(shuffle=True, write_csv=False)
        assert (
            test_sentences_rows_iob2 == sentences_rows_iob2
        ), f"ERROR! test_sentences_rows_iob2 = {test_sentences_rows_iob2} != {sentences_rows_iob2}"

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
        self.formatter_pretokenized.dataset_path = resource_filename(
            "nerblackbox", f"tests/test_data/formatted_data"
        )
        test_dfs = self.formatter_pretokenized.resplit_data(write_csv=False)
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
