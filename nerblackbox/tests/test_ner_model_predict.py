import pytest
from typing import Dict, List, Any, Tuple, Union
from nerblackbox.modules.ner_training.ner_model_predict import (
    get_tags_on_words_between_special_tokens,
    restore_unknown_tokens,
    restore_annotation_scheme_consistency,
    merge_tokens_to_entities,
)


"""
import json
from argparse import Namespace

from nerblackbox.modules.ner_training.ner_model_predict import NerModelPredict


class TestNerModelPredict:

    ner_model_predict = NerModelPredict(
        hparams=Namespace(**{
            "pretrained_model_name": "af-ai-center/bert-base-swedish-uncased",
            "uncased": True,
            "max_seq_length": 64,
            "tag_list": json.dumps(["PER"])
        })
    )
"""


class TestNerModelPredictStatic:
    @pytest.mark.parametrize(
        "tokens, " "example_token_predictions, " "example_word_predictions",
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
        ],
    )
    def test_get_tags_on_words_between_special_tokens(
        self,
        tokens: List[str],
        example_token_predictions: List[Any],
        example_word_predictions: Dict[str, List[Any]],
    ):
        test_example_word_predictions = get_tags_on_words_between_special_tokens(
            tokens, example_token_predictions
        )
        assert test_example_word_predictions == example_word_predictions, (
            f"test_example_word_predictions = "
            f"{test_example_word_predictions} != "
            f"{example_word_predictions}"
        )

    ####################################################################################################################
    @pytest.mark.parametrize(
        "example_word_predictions, " "example, " "example_word_predictions_restored",
        [
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
        ],
    )
    def test_restore_unknown_tokens(
        self,
        example_word_predictions: List[Tuple[Union[str, Any], ...]],
        example: str,
        example_word_predictions_restored: List[Dict[str, Any]],
    ):
        test_example_word_predictions_restored = restore_unknown_tokens(
            example_word_predictions,
            example,
            verbose=True,
        )
        assert (
            test_example_word_predictions_restored == example_word_predictions_restored
        ), (
            f"test_example_word_predictions_restored = "
            f"{test_example_word_predictions_restored} != "
            f"{example_word_predictions_restored}"
        )

    ####################################################################################################################
    @pytest.mark.parametrize(
        "example_word_predictions," "example_word_predictions_restored",
        [
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "I-ORG",
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
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
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
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "I-PER",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "O",
                    },
                ],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "B-PER",
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
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "I-PER",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "I-PER",
                    },
                ],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "B-PER",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "B-PER",
                    },
                ],
            ),
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "I-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "I-PER",
                    },
                    {
                        "char_start": "25",
                        "char_end": "26",
                        "token": "i",
                        "tag": "I-PER",
                    },
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "B-PER",
                    },
                ],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "B-PER",
                    },
                    {
                        "char_start": "25",
                        "char_end": "26",
                        "token": "i",
                        "tag": "I-PER",
                    },
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "B-PER",
                    },
                ],
            ),
            (
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
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "PER"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "PER",
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
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "PER"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "PER",
                    },
                ],
            ),
        ],
    )
    def test_restore_annotatione_scheme_consistency(
        self,
        example_word_predictions: List[Dict[str, Any]],
        example_word_predictions_restored: List[Dict[str, Any]],
    ):
        test_example_word_predictions_restored = restore_annotation_scheme_consistency(
            example_word_predictions
        )
        assert (
            test_example_word_predictions_restored == example_word_predictions_restored
        ), (
            f"test_example_word_predictions_restored = "
            f"{test_example_word_predictions_restored} != "
            f"{example_word_predictions_restored}"
        )

    ####################################################################################################################
    @pytest.mark.parametrize(
        "example_word_predictions," "example, " "example_word_predictions_merged",
        [
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
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
                "arbetsförmedlingen finns i stockholm",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                ],
            ),
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "B-PER",
                    },
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "O",
                    },
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
                        "tag": "PER",
                    },
                ],
            ),
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                    {
                        "char_start": "19",
                        "char_end": "24",
                        "token": "finns",
                        "tag": "B-PER",
                    },
                    {
                        "char_start": "25",
                        "char_end": "26",
                        "token": "i",
                        "tag": "I-PER",
                    },
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "B-PER",
                    },
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
                        "char_end": "26",
                        "token": "finns i",
                        "tag": "PER",
                    },
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "PER",
                    },
                ],
            ),
            (
                [
                    {
                        "char_start": "0",
                        "char_end": "8",
                        "token": "annotera",
                        "tag": "O",
                    },
                    {"char_start": "9", "char_end": "12", "token": "den", "tag": "ORG"},
                    {"char_start": "13", "char_end": "16", "token": "här", "tag": "O"},
                    {
                        "char_start": "17",
                        "char_end": "23",
                        "token": "texten",
                        "tag": "O",
                    },
                ],
                "annotera den här texten",
                [
                    {
                        "char_start": "9",
                        "char_end": "12",
                        "token": "den",
                        "tag": "ORG",
                    },
                ],
            ),
        ],
    )
    def test_merge_tokens_to_entities(
        self,
        example_word_predictions: List[Dict[str, Any]],
        example: str,
        example_word_predictions_merged: List[Dict[str, Any]],
    ):
        test_example_word_predictions_merged = merge_tokens_to_entities(
            example_word_predictions, example
        )
        assert (
            test_example_word_predictions_merged == example_word_predictions_merged
        ), (
            f"test_example_word_predictions_merged = "
            f"{test_example_word_predictions_merged} != "
            f"{example_word_predictions_merged}"
        )
