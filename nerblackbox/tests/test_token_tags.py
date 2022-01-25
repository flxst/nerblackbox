import pytest
from typing import Dict, List, Any
from nerblackbox.modules.ner_training.annotation_tags.token_tags import TokenTags


class TestTokenTags:
    ####################################################################################################################
    @pytest.mark.parametrize(
        "annotation_schemes," "example_word_predictions," "passes",
        [
            # 1. plain
            (
                ["plain"],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                ],
                True,
            ),
            # 2. bio/bilou
            (
                ["bio", "bilou"],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "B-ORG",
                    },
                ],
                True,
            ),
            # 3. plain/bio/bilou
            (
                ["bio", "bilou"],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "O",
                    },
                ],
                True,
            ),
            # 4. fail
            (
                ["bio", "bilou"],
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "ORG",
                    },
                ],
                False,
            ),
        ],
    )
    def test_assert_scheme_consistency(
        self,
        annotation_schemes: List[str],
        example_word_predictions: List[Dict[str, Any]],
        passes: bool,
    ):
        for annotation_scheme in annotation_schemes:
            if passes:
                TokenTags(example_word_predictions, scheme=annotation_scheme)
            else:
                with pytest.raises(Exception):
                    TokenTags(example_word_predictions, scheme=annotation_scheme)

    ####################################################################################################################
    @pytest.mark.parametrize(
        "annotation_scheme,"
        "example_word_predictions,"
        "example_word_predictions_restored",
        [
            # 1. plain: 2 single-token tags
            (
                "plain",
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
            # 2. bio: 1 single-token tag
            (
                "bio",
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
            # 3. bio: 2 single-token tags
            (
                "bio",
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
            # 4. bio: 3 single-token tags
            (
                "bio",
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
            # 5. bio: 2 single-token tags, 1 multiple-token tags
            (
                "bio",
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
            # 6. bilou: 2 single-token tags, 1 multiple-token tags
            (
                "bilou",
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
                        "tag": "U-ORG",
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
                        "tag": "L-PER",
                    },
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "U-PER",
                    },
                ],
            ),
        ],
    )
    def test_restore_annotatione_scheme_consistency(
        self,
        annotation_scheme: str,
        example_word_predictions: List[Dict[str, Any]],
        example_word_predictions_restored: List[Dict[str, Any]],
    ):
        token_tags = TokenTags(example_word_predictions, scheme=annotation_scheme)
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
        "scheme, "
        "example_word_predictions,"
        "example, "
        "example_word_predictions_merged",
        [
            # 1. BIO: 1 single word entity
            (
                "bio",
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
            # 2. BIO: 2 single word entities
            (
                "bio",
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
            # 3. BIO: 2 single word entities + 1 multiple word entity
            (
                "bio",
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
            # 4. BIO: 1 single word entities + 1 multiple word entity + 1 "lost" token
            (
                "bio",
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
            # 5. PLAIN: 1 single word entity
            (
                "plain",
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
            # 5. PLAIN: 1 multiple word entity
            (
                "plain",
                [
                    {
                        "char_start": "0",
                        "char_end": "8",
                        "token": "annotera",
                        "tag": "O",
                    },
                    {"char_start": "9", "char_end": "12", "token": "den", "tag": "ORG"},
                    {
                        "char_start": "13",
                        "char_end": "16",
                        "token": "här",
                        "tag": "ORG",
                    },
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
                        "char_end": "16",
                        "token": "den här",
                        "tag": "ORG",
                    },
                ],
            ),
            # 6. BILOU: 2 single word entities + 1 multiple word entity
            (
                "bilou",
                [
                    {
                        "char_start": "0",
                        "char_end": "18",
                        "token": "arbetsförmedlingen",
                        "tag": "U-ORG",
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
                        "tag": "L-PER",
                    },
                    {
                        "char_start": "27",
                        "char_end": "36",
                        "token": "stockholm",
                        "tag": "U-PER",
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
        ],
    )
    def test_merge_tokens_to_entities(
        self,
        scheme: str,
        example_word_predictions: List[Dict[str, Any]],
        example: str,
        example_word_predictions_merged: List[Dict[str, Any]],
    ):
        token_tags = TokenTags(example_word_predictions, scheme=scheme)
        token_tags.merge_tokens_to_entities(original_text=example, verbose=True)
        test_example_word_predictions_merged = token_tags.as_list()
        assert (
            test_example_word_predictions_merged == example_word_predictions_merged
        ), (
            f"test_example_word_predictions_merged = "
            f"{test_example_word_predictions_merged} != "
            f"{example_word_predictions_merged}"
        )
