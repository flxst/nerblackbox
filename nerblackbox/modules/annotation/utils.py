from typing import List, Tuple
from nerblackbox.modules.annotation.colors import get_label_color
import json


def extract_labels(_input_file: str) -> List[Tuple[str, str]]:
    """
    extract labels present in annotated text (nerblackbox format) in input_file

    Args:
        _input_file: e.g. 'batch_1.jsonl'

    Returns:
        labels: e.g. [("PI", "#000000")]
    """
    with open(_input_file, "r") as f:
        input_lines = [json.loads(line) for line in f]

    all_labels = list()
    for input_line in input_lines:
        single_labels = list(set([tag["tag"] for tag in input_line["tags"]]))
        all_labels = list(set(all_labels + single_labels))

    labels = [(label_name, get_label_color(i)) for i, label_name in enumerate(all_labels)]
    return labels
