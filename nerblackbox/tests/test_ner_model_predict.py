import pytest
from typing import Dict, List, Any, Tuple, Union
from nerblackbox.modules.ner_training.ner_model_predict import \
    get_tags_on_words_between_special_tokens, restore_unknown_tokens, restore_annotation_scheme_consistency


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
                {
                    "internal": [
                        ("arbetsförmedl", "ORG"),
                        ("##ingen", "ORG"),
                        ("finns", "O"),
                        ("i", "O"),
                        ("stockholm", "O"),
                    ],
                    "external": [
                        ("arbetsförmedlingen", "ORG"),
                        ("finns", "O"),
                        ("i", "O"),
                        ("stockholm", "O"),
                    ],
                },
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
        assert test_example_word_predictions.keys() == example_word_predictions.keys()
        for k in test_example_word_predictions.keys():
            assert test_example_word_predictions[k] == example_word_predictions[k], \
                f"key = {k}: test_example_word_predictions = " \
                f"{test_example_word_predictions[k]} != " \
                f"{example_word_predictions[k]}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "example_word_predictions_external, " "example, " "example_word_predictions_external_restored",
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
                    {"char_start":  "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
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
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
                ]
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
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
                ],
            ),
        ],
    )
    def test_restore_unknown_tokens(
            self,
            example_word_predictions_external: List[Tuple[Union[str, Any], ...]],
            example: str,
            example_word_predictions_external_restored: List[Dict[str, Any]],
    ):
        test_example_word_predictions_external_restored = restore_unknown_tokens(
            example_word_predictions_external,
            example,
            verbose=True,
        )
        assert test_example_word_predictions_external_restored == example_word_predictions_external_restored, \
            f"test_example_word_predictions_external_restored = " \
            f"{test_example_word_predictions_external_restored} != " \
            f"{example_word_predictions_external_restored}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "example_word_predictions_external," "example_word_predictions_external_restored",
        [
            (
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "I-ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
                ],
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
                ],
            ),
            (
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "I-PER"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
                ],
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "B-PER"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "O"},
                ],
            ),
            (
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "I-ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "I-PER"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "I-PER"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-PER"},
                ],
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "B-PER"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "I-PER"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-PER"},
                ],
            ),
            (
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "PER"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "PER"},
                ],
                [
                    {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
                    {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
                    {"char_start": "25", "char_end": "26", "token": "i", "tag": "PER"},
                    {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "PER"},
                ],
            ),
        ],
    )
    def test_restore_annotatione_scheme_consistency(
            self,
            example_word_predictions_external: List[Dict[str, Any]],
            example_word_predictions_external_restored: List[Dict[str, Any]],
    ):
        test_example_word_predictions_external_restored = \
            restore_annotation_scheme_consistency(example_word_predictions_external)
        assert test_example_word_predictions_external_restored == example_word_predictions_external_restored, \
            f"test_example_word_predictions_external_restored = " \
            f"{test_example_word_predictions_external_restored} != " \
            f"{example_word_predictions_external_restored}"
