import pytest
from typing import Dict, List, Any, Tuple, Union
import numpy as np

import torch

from nerblackbox.api.model import (
    EVALUATION_DICT,
    round_evaluation_dict,
    derive_annotation_scheme,
    turn_tensors_into_tag_probability_distributions,
    merge_slices_for_single_document,
    merge_subtoken_to_token_predictions,
    restore_unknown_tokens,
    assert_typing,
)


class TestModelStatic:
    @pytest.mark.parametrize(
        "evaluation_dict, rounded_decimals, evaluation_dict_rounded",
        [
            (
                {
                    "micro": {
                        "entity": {
                            "precision": 0.9847222222222223,
                            "recall": 0.9916083916083916,
                            "f1": 0.9881533101045297,
                            "precision_seqeval": 0.9833564493758669,
                            "recall_seqeval": 0.9916083916083916,
                            "f1_seqeval": 0.9874651810584958,
                        },
                    }
                },
                3,
                {
                    "micro": {
                        "entity": {
                            "precision": 0.985,
                            "recall": 0.992,
                            "f1": 0.988,
                            "precision_seqeval": 0.983,
                            "recall_seqeval": 0.992,
                            "f1_seqeval": 0.987,
                        }
                    }
                },
            ),
        ],
    )
    def test_derive_evaluation_dict(
        self,
        evaluation_dict: EVALUATION_DICT,
        rounded_decimals: int,
        evaluation_dict_rounded: EVALUATION_DICT,
    ):
        test_evaluation_dict_rounded = round_evaluation_dict(
            evaluation_dict, rounded_decimals
        )
        assert test_evaluation_dict_rounded == evaluation_dict_rounded, (
            f"ERROR! test_evaluation_dict_rounded = {test_evaluation_dict_rounded} "
            f"!= {evaluation_dict_rounded} = evaluation_dict_rounded"
        )

    @pytest.mark.parametrize(
        "id2label, error, annotation_scheme",
        [
            (
                {0: "O", 1: "PER"},
                False,
                "plain",
            ),
            (
                {0: "O", 1: "B-PER", 2: "I-PER"},
                False,
                "bio",
            ),
            (
                {0: "O", 1: "B-PER", 2: "I-PER", 3: "L-PER", 4: "U-PER"},
                False,
                "bilou",
            ),
            (
                {0: "O", 1: "A-PER", 2: "B-PER"},
                True,
                "",
            ),
            (
                {},
                True,
                "",
            ),
        ],
    )
    def test_derive_annotation_scheme(
        self, id2label: Dict[int, str], error: bool, annotation_scheme: str
    ):
        if error:
            with pytest.raises(Exception):
                _ = derive_annotation_scheme(id2label)
        else:
            test_annotation_scheme = derive_annotation_scheme(id2label)
            assert (
                test_annotation_scheme == annotation_scheme
            ), f"ERROR! test_annotation_scheme = {test_annotation_scheme} != {annotation_scheme} = annotation_scheme"

    @pytest.mark.parametrize(
        "annotation_classes, outputs, predictions",
        [
            (
                ["O", "B-PER", "I-PER"],
                torch.tensor([[[0.01, 3, 0.01]]]),
                [[{"O": 0.0457, "B-PER": 0.9086, "I-PER": 0.0457}]],
            ),
        ],
    )
    def test_turn_tensors_into_tag_probability_distributions(
        self,
        annotation_classes: List[str],
        outputs: torch.Tensor,
        predictions: List[List[Dict[str, float]]],
    ):
        test_predictions = turn_tensors_into_tag_probability_distributions(
            annotation_classes, outputs
        )
        for list1, list2 in zip(test_predictions, predictions):
            for dict1, dict2 in zip(list1, list2):
                assert set(dict1.keys()) == set(
                    dict2.keys()
                ), f"ERROR! test keys = {set(dict1.keys())} != {set(dict2.keys())} = keys"
                for k in dict1:
                    assert (
                        np.absolute(dict1[k] - dict2[k]) < 0.0001
                    ), f"ERROR! test_predictions = {dict1} != {dict2} = predictions"

    @pytest.mark.parametrize(
        "list_slices, list_documents",
        [
            # 1. tokens
            (
                [["[CLS]", "this", "is", "one", "slice", "[SEP]"]],
                ["[CLS]", "this", "is", "one", "slice", "[SEP]"],
            ),
            (
                [
                    ["[CLS]", "this", "is", "one", "slice", "[SEP]"],
                    ["[CLS]", "and", "a", "second", "one", "[SEP]"],
                ],
                [
                    "[CLS]",
                    "this",
                    "is",
                    "one",
                    "slice",
                    "and",
                    "a",
                    "second",
                    "one",
                    "[SEP]",
                ],
            ),
            (
                [
                    ["[CLS]", "slice", "1", "[SEP]"],
                    ["[CLS]", "slice", "2", "[SEP]"],
                    ["[CLS]", "slice", "3", "[SEP]"],
                ],
                ["[CLS]", "slice", "1", "slice", "2", "slice", "3", "[SEP]"],
            ),
            # 2. predictions
            (
                [
                    ["O", "B-PER", "I-PER", "[S]"],
                    ["[S]", "B-LOC", "I-LOC", "[SEP]"],
                    ["[S]", "B-ORG", "I-ORG", "O"],
                ],
                ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "O"],
            ),
            # 3. predictions with proba = True
            (
                [
                    [
                        {"O": 0.1, "PER": 0.9},
                        {"O": 0.1, "PER": 0.9},
                        {"O": 1.1, "PER": 1.1},
                    ],
                    [
                        {"O": 1.1, "PER": 1.1},
                        {"O": 0.1, "PER": 0.9},
                        {"O": 0.1, "PER": 0.9},
                    ],
                ],
                [
                    {"O": 0.1, "PER": 0.9},
                    {"O": 0.1, "PER": 0.9},
                    {"O": 0.1, "PER": 0.9},
                    {"O": 0.1, "PER": 0.9},
                ],
            ),
        ],
    )
    def test_merge_slices_for_single_document(
        self,
        list_slices: List[List[Union[str, Dict[str, float]]]],
        list_documents: List[Union[str, Dict[str, float]]],
    ):
        test_list_documents = merge_slices_for_single_document(list_slices)
        assert (
            test_list_documents == list_documents
        ), f"ERROR! test_list_documents = {test_list_documents} != {list_documents} = list_documents"

    @pytest.mark.parametrize(
        "tokens, predictions, token_predictions",
        [
            (
                [
                    "[CLS]",
                    "arbetsförmedl",
                    "##ingen",
                    "finns",
                    "i",
                    "stockholm",
                    "[SEP]",
                    "[PAD]",
                ],
                [
                    "[S]",
                    "ORG",
                    "ORG",
                    "O",
                    "O",
                    "O",
                    "[S]",
                    "[S]",
                ],
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("finns", "O"),
                    ("i", "O"),
                    ("stockholm", "O"),
                ],
            ),
            (
                [
                    "[CLS]",
                    "arbetsförmedl",
                    "##ingen",
                    "finns",
                    "i",
                    "stockholm",
                    "[SEP]",
                    "[PAD]",
                ],
                [
                    "[S]",
                    "ORG",
                    "PER",
                    "O",
                    "O",
                    "O",
                    "[S]",
                    "[S]",
                ],
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("finns", "O"),
                    ("i", "O"),
                    ("stockholm", "O"),
                ],
            ),
            (
                [
                    "[CLS]",
                    "1996",
                    "-",
                    "08",
                    "-",
                    "30",
                    "[PAD]",
                ],
                [
                    "[S]",
                    "ORG",
                    "ORG",
                    "O",
                    "ORG",
                    "O",
                    "[S]",
                ],
                [
                    ("1996", "ORG"),
                    ("-", "ORG"),
                    ("08", "O"),
                    ("-", "ORG"),
                    ("30", "O"),
                ],
            ),
        ],
    )
    def test_merge_subtoken_to_token_predictions(
        self,
        tokens: List[str],
        predictions: List[Union[str, Dict[str, float]]],
        token_predictions: List[Tuple[Union[str, Dict[str, float]]]],
    ):
        test_token_predictions = merge_subtoken_to_token_predictions(
            tokens, predictions
        )
        assert test_token_predictions == token_predictions, (
            f"test_token_predictions = "
            f"{test_token_predictions} != "
            f"{token_predictions}"
        )

    ####################################################################################################################
    @pytest.mark.parametrize(
        "word_predictions, input_text, word_predictions_restored",
        [
            # EXAMPLE 1 #######################
            (
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("[UNK]", "O"),
                    ("i", "O"),
                    ("stockholm", "O"),
                ],
                "arbetsförmedlingen finns i stockholm",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "O",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "O",
                    },
                ],
            ),
            # EXAMPLE 2 #######################
            (
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("finns", "O"),
                    ("i", "O"),
                    ("[UNK]", "O"),
                ],
                "arbetsförmedlingen finns i stockholm",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "O",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "O",
                    },
                ],
            ),
            # EXAMPLE 3 #######################
            (
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("[UNK]", "O"),
                    ("i", "O"),
                    ("[UNK]", "O"),
                ],
                "arbetsförmedlingen finns i stockholm",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "O",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "O",
                    },
                ],
            ),
            # EXAMPLE 4 #######################
            (
                [
                    ("Du", "O"),
                    ("behöver", "O"),
                    ("[UNK]", "O"),
                    ("kunna", "O"),
                    ("programmera", "B-SKILL"),
                ],
                "Du behöver ✓ kunna programmera",
                [
                    {
                        "char_start": "0",
                        "char_end": "2",
                        "token": "Du",
                        "tag": "O",
                    },
                    {
                        "char_start": "3",
                        "char_end": "10",
                        "token": "behöver",
                        "tag": "O",
                    },
                    {
                        "char_start": "11",
                        "char_end": "12",
                        "token": "✓",
                        "tag": "O",
                    },
                    {
                        "char_start": "13",
                        "char_end": "18",
                        "token": "kunna",
                        "tag": "O",
                    },
                    {
                        "char_start": "19",
                        "char_end": "30",
                        "token": "programmera",
                        "tag": "B-SKILL",
                    },
                ],
            ),
            # EXAMPLE 5 #######################
            (
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("[UNK]", "O"),
                    ("i", "O"),
                    ("[UNK]", "O"),
                ],
                "arbetsförmedlingen íi i í",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "21",
                        "token": "íi",
                        "tag": "O",
                    },
                    {"char_start": "22", "char_end": "23", "token": "i", "tag": "O"},
                    {
                        "char_start": "24",
                        "char_end": "25",
                        "token": "í",
                        "tag": "O",
                    },
                ],
            ),
            # EXAMPLE 6 #######################
            (
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("[UNK]", "O"),
                    (".", "O"),
                    ("i", "O"),
                    ("[UNK]", "O"),
                ],
                "arbetsförmedlingen íi. i í",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "21",
                        "token": "íi",
                        "tag": "O",
                    },
                    {"char_start": "21", "char_end": "22", "token": ".", "tag": "O"},
                    {"char_start": "23", "char_end": "24", "token": "i", "tag": "O"},
                    {
                        "char_start": "25",
                        "char_end": "26",
                        "token": "í",
                        "tag": "O",
                    },
                ],
            ),
            # EXAMPLE 7 #######################
            (
                [
                    ("arbetsförmedlingen", "ORG"),
                    ("[UNK]", "O"),
                    ("!", "O"),
                    ("i", "O"),
                    ("[UNK]", "O"),
                ],
                "arbetsförmedlingen íi! i í",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "21",
                        "token": "íi",
                        "tag": "O",
                    },
                    {"char_start": "21", "char_end": "22", "token": "!", "tag": "O"},
                    {"char_start": "23", "char_end": "24", "token": "i", "tag": "O"},
                    {
                        "char_start": "25",
                        "char_end": "26",
                        "token": "í",
                        "tag": "O",
                    },
                ],
            ),
            # EXAMPLE 8 #######################
            (
                [
                    ("medarbetare", "O"),
                    ("i", "O"),
                    ("Stockholm", "O"),
                    (".", "O"),
                    ("[UNK]", "O"),
                    ("[NEWLINE]", "O"),
                    ("[NEWLINE]", "O"),
                    ("På", "O"),
                    ("Företaget", "O"),
                ],
                "medarbetare i Stockholm. ‍‍👨👩[NEWLINE][NEWLINE]På Företaget",
                [
                    {
                        "char_start": "0",
                        "char_end": "11",
                        "token": "medarbetare",
                        "tag": "O",
                    },
                    {
                        "char_start": "12",
                        "char_end": "13",
                        "token": "i",
                        "tag": "O",
                    },
                    {
                        "char_start": "14",
                        "char_end": "23",
                        "token": "Stockholm",
                        "tag": "O",
                    },
                    {
                        "char_start": "23",
                        "char_end": "24",
                        "token": ".",
                        "tag": "O",
                    },
                    {
                        "char_start": "25",
                        "char_end": "29",
                        "token": "‍‍👨👩",
                        "tag": "O",
                    },
                    {
                        "char_start": "29",
                        "char_end": "38",
                        "token": "[NEWLINE]",
                        "tag": "O",
                    },
                    {
                        "char_start": "38",
                        "char_end": "47",
                        "token": "[NEWLINE]",
                        "tag": "O",
                    },
                    {
                        "char_start": "47",
                        "char_end": "49",
                        "token": "På",
                        "tag": "O",
                    },
                    {
                        "char_start": "50",
                        "char_end": "59",
                        "token": "Företaget",
                        "tag": "O",
                    },
                ],
            ),
            #
        ],
    )
    def test_restore_unknown_tokens(
        self,
        word_predictions: List[
            Tuple[str, Union[str, Dict[str, float]]]
        ],  # List[Tuple[Union[str, Any], ...]],  # #
        input_text: str,
        word_predictions_restored: List[Dict[str, Any]],
    ):
        test_word_predictions_restored = restore_unknown_tokens(
            word_predictions,
            input_text,
            verbose=True,
        )
        assert test_word_predictions_restored == word_predictions_restored, (
            f"test_word_predictions_restored = "
            f"{test_word_predictions_restored} != "
            f"{word_predictions_restored}"
        )

    @pytest.mark.parametrize(
        "word_predictions, word_predictions_str",
        [
            (
                [
                    {
                        "char_start": 0,
                        "char_end": 18,
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": 19,
                        "char_end": 24,
                        "token": "finns",
                        "tag": "O",
                    },
                ],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "O",
                    },
                ],
            )
        ],
    )
    def test_assert_typing(
        self,
        word_predictions: List[Dict[str, Any]],
        word_predictions_str: List[Dict[str, str]],
    ):
        test_word_predictions_str = assert_typing(
            word_predictions,
        )
        assert test_word_predictions_str == word_predictions_str, (
            f"test_word_predictions_str = "
            f"{test_word_predictions_str} != "
            f"{word_predictions_str}"
        )
