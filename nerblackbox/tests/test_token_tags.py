import pytest
from typing import Dict, List, Any
from nerblackbox.modules.ner_training.annotation_tags.token_tags import TokenTags


class TestTokenTags:
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
        token_tags = TokenTags(example_word_predictions)
        token_tags.restore_annotation_scheme_consistency()
        test_example_word_predictions_restored = token_tags.as_list()
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
        token_tags = TokenTags(example_word_predictions)
        token_tags.merge_tokens_to_entities(original_text=example, verbose=True)
        test_example_word_predictions_merged = token_tags.as_list()
        assert (
                test_example_word_predictions_merged == example_word_predictions_merged
        ), (
            f"test_example_word_predictions_merged = "
            f"{test_example_word_predictions_merged} != "
            f"{example_word_predictions_merged}"
        )
