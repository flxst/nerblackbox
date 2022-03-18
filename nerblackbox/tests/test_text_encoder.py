import pytest
from typing import List, Dict, Tuple, Union
from nerblackbox.modules.ner_training.data_preprocessing.text_encoder import TextEncoder

Encodings = List[Tuple[int, str, str]]
Predictions = List[Dict[str, Union[str, Dict]]]


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestTextEncoder:

    ####################################################################################################################
    @pytest.mark.parametrize(
        "encodings_mapping, text_list, predictions_list, predictions_encoded_list",
        [
            # Example 1: level = entity
            (
                {"\n": "[NEWLINE]", "\t": "[TAB]"},
                ["an\n example"],
                [
                    [
                        {
                            "char_start": "4",
                            "char_end": "11",
                            "token": "example",
                            "tag": "TAG",
                        }
                    ]
                ],
                [
                    [
                        {
                            "char_start": "12",
                            "char_end": "19",
                            "token": "example",
                            "tag": "TAG",
                        }
                    ]
                ],
            ),
            # Example 2: level = word
            (
                {"\n": "[NEWLINE]", "\t": "[TAB]"},
                ["an\n example"],
                [
                    [
                        {"char_start": "0", "char_end": "2", "token": "an", "tag": "O"},
                        {"char_start": "2", "char_end": "3", "token": "\n", "tag": "O"},
                        {
                            "char_start": "4",
                            "char_end": "11",
                            "token": "example",
                            "tag": "TAG",
                        },
                    ]
                ],
                [
                    [
                        {"char_start": "0", "char_end": "2", "token": "an", "tag": "O"},
                        {
                            "char_start": "2",
                            "char_end": "11",
                            "token": "[NEWLINE]",
                            "tag": "O",
                        },
                        {
                            "char_start": "12",
                            "char_end": "19",
                            "token": "example",
                            "tag": "TAG",
                        },
                    ]
                ],
            ),
        ],
    )
    def tests(
        self,
        encodings_mapping: Dict[str, str],
        text_list: List[str],
        predictions_list: List[Predictions],
        predictions_encoded_list: List[Predictions],
    ) -> None:
        text_encoder = TextEncoder(encodings_mapping)

        text_encoded_list, encodings_list = text_encoder.encode(text_list)
        test_text_list, test_predictions_list = text_encoder.decode(
            text_encoded_list, encodings_list, predictions_encoded_list
        )
        assert (
            test_text_list == text_list
        ), f"ERROR! test_text_list = {test_text_list} != text_list = {text_list}"
        assert (
            test_predictions_list == predictions_list
        ), f"ERROR! test_predictions_list = {test_predictions_list} != predictions_list = {predictions_list}"
