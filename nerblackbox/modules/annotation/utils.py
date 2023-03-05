from typing import List, Tuple, Dict, Any
from nerblackbox.modules.annotation.colors import get_label_color
from nerblackbox.modules.annotation.io import read_jsonl


def extract_labels(_input_file: str) -> List[Tuple[str, str]]:
    """
    extract labels present in annotated text (nerblackbox format) in input_file

    Args:
        _input_file: e.g. 'batch_1.jsonl'

    Returns:
        labels: e.g. [("PI", "#000000")]
    """
    input_lines = read_jsonl(_input_file)
    return _extract_labels(input_lines)


def _extract_labels(_input_lines: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    extract labels present in annotated text (nerblackbox format) in _input_lines

    Args:
        _input_lines: e.g.
            [
                {
                    "text": "\n2020-05-20 John Doe pratar.",
                    "tags": [
                        {"char_start": 1, "char_end": 11, "token": "2020-05-20", "tag": "PI"},
                        {"char_start": 12, "char_end": 20, "token": "John Doe", "tag": "PI"},
                        ]
                }
            ]

    Returns:
        labels: e.g. [("PI", "#000000")]
    """

    all_labels = list()
    for input_line in _input_lines:
        single_labels = list(set([tag["tag"] for tag in input_line["tags"]]))
        all_labels = list(set(all_labels + single_labels))

    labels = [(label_name, get_label_color(i)) for i, label_name in enumerate(all_labels)]
    return labels
