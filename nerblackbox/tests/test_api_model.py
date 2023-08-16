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
        "tokens, predictions, tokenizer_type, token_predictions",
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
                "WordPiece",
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
                "WordPiece",
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
                "WordPiece",
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
        tokenizer_type: str,
        token_predictions: List[Tuple[Union[str, Dict[str, float]]]],
    ):
        tokenizer_special = ["[CLS]", "[SEP]", "[PAD]"] if tokenizer_type == "WordPiece" else ['</s>', '<s>', '<pad>']
        test_token_predictions = merge_subtoken_to_token_predictions(
            tokens, predictions, tokenizer_special=tokenizer_special, tokenizer_type=tokenizer_type
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
            # EXAMPLE 9 #######################
            (
                    [
                        ('diese', 'O'), ('großraumwagen', 'O'), ('2', 'O'), ('.', 'O'), ('klasse', 'O'), ('mit', 'O'),
                        ('62', 'O'), ('sitzplatzen', 'O'), ('in', 'O'), ('vis', 'O'), ('-', 'O'), ('a', 'O'), ('-', 'O'),
                        ('vis', 'O'), ('-', 'O'), ('anordnung', 'O'), ('vom', 'O')
                    ],
                    "diese großraumwagen 2. klasse mit 62 sitzplätzen in vis-à-vis-anordnung vom",
                    [
                        {'char_start': '0', 'char_end': '5', 'token': 'diese', 'tag': 'O'},
                        {'char_start': '6', 'char_end': '19', 'token': 'großraumwagen', 'tag': 'O'},
                        {'char_start': '20', 'char_end': '21', 'token': '2', 'tag': 'O'},
                        {'char_start': '21', 'char_end': '22', 'token': '.', 'tag': 'O'},
                        {'char_start': '23', 'char_end': '29', 'token': 'klasse', 'tag': 'O'},
                        {'char_start': '30', 'char_end': '33', 'token': 'mit', 'tag': 'O'},
                        {'char_start': '34', 'char_end': '36', 'token': '62', 'tag': 'O'},
                        {'char_start': '37', 'char_end': '48', 'token': 'sitzplätzen', 'tag': 'O'},
                        {'char_start': '49', 'char_end': '51', 'token': 'in', 'tag': 'O'},
                        {'char_start': '52', 'char_end': '55', 'token': 'vis', 'tag': 'O'},
                        {'char_start': '55', 'char_end': '56', 'token': '-', 'tag': 'O'},
                        {'char_start': '56', 'char_end': '57', 'token': 'à', 'tag': 'O'},
                        {'char_start': '57', 'char_end': '58', 'token': '-', 'tag': 'O'},
                        {'char_start': '58', 'char_end': '61', 'token': 'vis', 'tag': 'O'},
                        {'char_start': '61', 'char_end': '62', 'token': '-', 'tag': 'O'},
                        {'char_start': '62', 'char_end': '71', 'token': 'anordnung', 'tag': 'O'},
                        {'char_start': '72', 'char_end': '75', 'token': 'vom', 'tag': 'O'},
                    ],
            ),
            # EXAMPLE 10 #######################
            (
                    [
                        ('der', 'O'), ('titel', 'O'), (',', 'O'), ('den', 'O'), ('sie', 'O'), ('mit', 'O'),
                        ('ihrer', 'O'),
                        ('erhohung', 'O'), ('erhielt', 'O'), (',', 'O'), ('lautete', 'O'), ('yi', 'B-PER'),
                        ('guifei', 'O'),
                        ('(', 'O'), ('[UNK]', 'O'), ('[UNK]', 'O'), ('[UNK]', 'O'), (')', 'O'),
                    ],
                    "der titel , den sie mit ihrer erhöhung erhielt , lautete yi guifei ( 懿貴妃 )",
                    [
                        {'char_start': '0', 'char_end': '3', 'token': 'der', 'tag': 'O'},
                        {'char_start': '4', 'char_end': '9', 'token': 'titel', 'tag': 'O'},
                        {'char_start': '10', 'char_end': '11', 'token': ',', 'tag': 'O'},
                        {'char_start': '12', 'char_end': '15', 'token': 'den', 'tag': 'O'},
                        {'char_start': '16', 'char_end': '19', 'token': 'sie', 'tag': 'O'},
                        {'char_start': '20', 'char_end': '23', 'token': 'mit', 'tag': 'O'},
                        {'char_start': '24', 'char_end': '29', 'token': 'ihrer', 'tag': 'O'},
                        {'char_start': '30', 'char_end': '38', 'token': 'erhöhung', 'tag': 'O'},
                        {'char_start': '39', 'char_end': '46', 'token': 'erhielt', 'tag': 'O'},
                        {'char_start': '47', 'char_end': '48', 'token': ',', 'tag': 'O'},
                        {'char_start': '49', 'char_end': '56', 'token': 'lautete', 'tag': 'O'},
                        {'char_start': '57', 'char_end': '59', 'token': 'yi', 'tag': 'B-PER'},
                        {'char_start': '60', 'char_end': '66', 'token': 'guifei', 'tag': 'O'},
                        {'char_start': '67', 'char_end': '68', 'token': '(', 'tag': 'O'},
                        {'char_start': '69', 'char_end': '72', 'token': '懿貴妃', 'tag': 'O'},  # word restored.
                        {'char_start': '73', 'char_end': '74', 'token': ')', 'tag': 'O'},
                    ]
            ),
            # EXAMPLE 11 #######################
            (
                    [
                        ('auf', 'O'), ('dem', 'O'), ('berg', 'O'), ('gibt', 'O'), ('es', 'O'),
                        ('zwei', 'O'), ('aussichtspavillions', 'O'), (',', 'O'), ('tongdae', 'B-LOC'), ('(', 'O'),
                        ('[UNK]', 'B-LOC'), ('[UNK]', 'I-LOC'), ('[UNK]', 'O'), (')', 'O'), ('und', 'O'),
                        ('hakpyollu', 'B-LOC'),
                        ('(', 'O'), ('[UNK]', 'B-LOC'), ('[UNK]', 'I-LOC'), ('[UNK]', 'I-LOC'), ('[UNK]', 'O'),
                        (')', 'O'), ('.', 'O')
                    ],
                    "auf dem berg gibt es zwei aussichtspavillions , tongdae ( 동대 東台 ) und hakpyŏllu ( 학별루 鶴別樓 ) .",
                    [
                        {'char_start': '0', 'char_end': '3', 'token': 'auf', 'tag': 'O'},
                        {'char_start': '4', 'char_end': '7', 'token': 'dem', 'tag': 'O'},
                        {'char_start': '8', 'char_end': '12', 'token': 'berg', 'tag': 'O'},
                        {'char_start': '13', 'char_end': '17', 'token': 'gibt', 'tag': 'O'},
                        {'char_start': '18', 'char_end': '20', 'token': 'es', 'tag': 'O'},
                        {'char_start': '21', 'char_end': '25', 'token': 'zwei', 'tag': 'O'},
                        {'char_start': '26', 'char_end': '45', 'token': 'aussichtspavillions', 'tag': 'O'},
                        {'char_start': '46', 'char_end': '47', 'token': ',', 'tag': 'O'},
                        {'char_start': '48', 'char_end': '55', 'token': 'tongdae', 'tag': 'B-LOC'},
                        {'char_start': '56', 'char_end': '57', 'token': '(', 'tag': 'O'},
                        {'char_start': '58', 'char_end': '60', 'token': '동대', 'tag': 'O'},
                        {'char_start': '61', 'char_end': '63', 'token': '東台', 'tag': 'O'},
                        {'char_start': '64', 'char_end': '65', 'token': ')', 'tag': 'O'},
                        {'char_start': '66', 'char_end': '69', 'token': 'und', 'tag': 'O'},
                        {'char_start': '70', 'char_end': '79', 'token': 'hakpyŏllu', 'tag': 'B-LOC'},
                        {'char_start': '80', 'char_end': '81', 'token': '(', 'tag': 'O'},
                        {'char_start': '82', 'char_end': '85', 'token': '학별루', 'tag': 'O'},
                        {'char_start': '86', 'char_end': '89', 'token': '鶴別樓', 'tag': 'O'},
                        {'char_start': '90', 'char_end': '91', 'token': ')', 'tag': 'O'},
                        {'char_start': '92', 'char_end': '93', 'token': '.', 'tag': 'O'},
                    ]
            ),
            # EXAMPLE 12 #######################
            (
                    [
                        ('The', 'O'), ('official', 'O'), ('Itar-Tass', 'B-ORG'), ('news', 'O'), ('agency', 'O'),
                        ('quoted', 'O'), ('Livshits', 'B-PER'), ('as', 'O'), ('telling', 'O'), ('parliamentary', 'O'),
                        ('deputies', 'O'), ('that', 'O'), ('RAO', 'B-ORG'), ('Norilsky', 'I-ORG'), ('Nikel', 'I-ORG'),
                        ('0NKEL.RUO', 'O'), ('had', 'O'), ('to', 'O'), ('pay', 'O'), ('its', 'O'), ('tax', 'O')
                    ],
                    "The official Itar-Tass news agency quoted Livshits as telling parliamentary deputies that RAO Norilsky Nikel 0#NKEL.RUO had to pay its tax",
                    [
                        {'char_start': '0', 'char_end': '3', 'token': 'The', 'tag': 'O'},
                        {'char_start': '4', 'char_end': '12', 'token': 'official', 'tag': 'O'},
                        {'char_start': '13', 'char_end': '22', 'token': 'Itar-Tass', 'tag': 'B-ORG'},
                        {'char_start': '23', 'char_end': '27', 'token': 'news', 'tag': 'O'},
                        {'char_start': '28', 'char_end': '34', 'token': 'agency', 'tag': 'O'},
                        {'char_start': '35', 'char_end': '41', 'token': 'quoted', 'tag': 'O'},
                        {'char_start': '42', 'char_end': '50', 'token': 'Livshits', 'tag': 'B-PER'},
                        {'char_start': '51', 'char_end': '53', 'token': 'as', 'tag': 'O'},
                        {'char_start': '54', 'char_end': '61', 'token': 'telling', 'tag': 'O'},
                        {'char_start': '62', 'char_end': '75', 'token': 'parliamentary', 'tag': 'O'},
                        {'char_start': '76', 'char_end': '84', 'token': 'deputies', 'tag': 'O'},
                        {'char_start': '85', 'char_end': '89', 'token': 'that', 'tag': 'O'},
                        {'char_start': '90', 'char_end': '93', 'token': 'RAO', 'tag': 'B-ORG'},
                        {'char_start': '94', 'char_end': '102', 'token': 'Norilsky', 'tag': 'I-ORG'},
                        {'char_start': '103', 'char_end': '108', 'token': 'Nikel', 'tag': 'I-ORG'},
                        {'char_start': '109', 'char_end': '119', 'token': '0#NKEL.RUO', 'tag': 'O'},
                        {'char_start': '120', 'char_end': '123', 'token': 'had', 'tag': 'O'},
                        {'char_start': '124', 'char_end': '126', 'token': 'to', 'tag': 'O'},
                        {'char_start': '127', 'char_end': '130', 'token': 'pay', 'tag': 'O'},
                        {'char_start': '131', 'char_end': '134', 'token': 'its', 'tag': 'O'},
                        {'char_start': '135', 'char_end': '138', 'token': 'tax', 'tag': 'O'},
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
