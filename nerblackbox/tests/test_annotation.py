import pytest
from typing import List, Dict, Any, Tuple

from nerblackbox.modules.annotation.colors import get_label_color
from nerblackbox.modules.annotation.utils import _extract_labels
from nerblackbox.modules.annotation.file_conversion import nerblackbox2labelstudio, labelstudio2nerblackbox
from nerblackbox.modules.annotation.file_conversion import nerblackbox2doccano, doccano2nerblackbox


class TestAnnotationColors:
    @pytest.mark.parametrize(
        "index, label_color",
        [
            (0, "#000000"),   # black
            (2, "#1F618D"),   # blue dark
            (18, "#000000"),  # black
            (20, "#1F618D"),  # blue dark
        ],
    )
    def test_get_label_color(self, index: int, label_color: str):
        test_label_color = get_label_color(index)
        assert test_label_color == label_color, f"ERROR! test_label_color = {test_label_color} != {label_color}"


class TestAnnotationUtils:
    @pytest.mark.parametrize(
        "_input_lines, labels",
        [
            (
                    [
                        {"text": "hello", "tags": []}
                    ],
                    [],
            ),
            (
                    [
                        {
                            "text": "\n2020-05-20 John Doe pratar.",
                            "tags": [
                                {"char_start": 1, "char_end": 11, "token": "2020-05-20", "tag": "PI"},
                                {"char_start": 12, "char_end": 20, "token": "John Doe", "tag": "PI"},
                                ]
                        }
                    ],
                    [("PI", "#000000")]
            ),
        ],
    )
    def test_extract_labels(self, _input_lines: List[Dict[str, Any]], labels: List[Tuple[str, str]]):
        test_labels = _extract_labels(_input_lines)
        assert test_labels == labels, f"ERROR! test_labels = {test_labels} != {labels}"


class TestAnnotationFileConversion:
    @pytest.mark.parametrize(
        "_input_lines, _output_lines",
        [
            (
                    # input_line
                    [
                        {
                            "text": "\n2020-05-20 John Doe pratar.",
                            "tags": [
                                {"char_start": 1, "char_end": 11, "token": "2020-05-20", "tag": "PI"},
                                {"char_start": 12, "char_end": 20, "token": "John Doe", "tag": "PI"},
                            ]
                        }
                    ],
                    # output_line
                    [
                        {
                            "data": {
                                "text": "\n2020-05-20 John Doe pratar.",
                            },
                            "annotations": [{
                                "result": [
                                    {
                                        "id": "0",
                                        "from_name": "label",
                                        "to_name": "text",
                                        "type": "labels",
                                        "value": {
                                            "start": 1,
                                            "end": 11,
                                            "text": "2020-05-20",
                                            "labels": ["PI"],
                                        },
                                    },
                                    {
                                        "id": "1",
                                        "from_name": "label",
                                        "to_name": "text",
                                        "type": "labels",
                                        "value": {
                                            "start": 12,
                                            "end": 20,
                                            "text": "John Doe",
                                            "labels": ["PI"],
                                        },
                                    },
                                ]
                            }]
                        }
                    ],
            ),
        ],
    )
    def test_nerblackbox2labelstudio(self, _input_lines: List[Dict[str, Any]], _output_lines: List[Dict[str, Any]]):
        test_output_lines = nerblackbox2labelstudio(_input_lines)
        assert test_output_lines == _output_lines, f"ERROR! test_output_lines = {test_output_lines} != {_output_lines}"

        test_input_lines = labelstudio2nerblackbox(_output_lines)
        assert test_input_lines == _input_lines, f"ERROR! test_input_lines = {test_input_lines} != {_input_lines}"

    @pytest.mark.parametrize(
        "_input_lines, _output_lines",
        [
            (
                    # input_line
                    [
                        {
                            "text": "\n2020-05-20 John Doe pratar.",
                            "tags": [
                                {"char_start": 1, "char_end": 11, "token": "2020-05-20", "tag": "PI"},
                                {"char_start": 12, "char_end": 20, "token": "John Doe", "tag": "PI"},
                            ]
                        }
                    ],
                    # output_line
                    [
                        {
                            "text": "\n2020-05-20 John Doe pratar.",
                            "label": [
                                [1, 11, "PI"],
                                [12, 20, "PI"],
                            ]
                        }
                    ],
            ),
        ],
    )
    def test_nerblackbox2doccano(self, _input_lines: List[Dict[str, Any]], _output_lines: List[Dict[str, Any]]):
        test_output_lines = nerblackbox2doccano(_input_lines)
        assert test_output_lines == _output_lines, f"ERROR! test_output_lines = {test_output_lines} != {_output_lines}"

        test_input_lines = doccano2nerblackbox(_output_lines)
        assert test_input_lines == _input_lines, f"ERROR! test_input_lines = {test_input_lines} != {_input_lines}"
