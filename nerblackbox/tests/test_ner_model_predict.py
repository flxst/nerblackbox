import pytest
from typing import Dict, List, Any, Tuple, Union
from nerblackbox.modules.ner_training.ner_model_predict import (
    get_tags_on_words_between_special_tokens,
    restore_unknown_tokens,
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
