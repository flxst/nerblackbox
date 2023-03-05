
import os
import json
from typing import List, Dict, Any


def read_jsonl(_input_file: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    read a jsonl file line by line and return content

    Args:
        _input_file: e.g. 'batch_1.jsonl'
        verbose: e.g. False

    Returns:
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
    """
    if verbose:
        print(f"> read input_file = {_input_file}")
    with open(_input_file, "r") as f:
        input_lines = [json.loads(line) for line in f]
    return input_lines


def write_jsonl(_output_file: str, _output_lines: List[Dict[str, Any]], verbose: bool = False) -> None:
    """
    write _output_lines to jsonl file line by line

    Args:
        _output_file: e.g. 'batch_1.jsonl'
        _output_lines: e.g.
            [
                {
                    "text": "\n2020-05-20 John Doe pratar.",
                    "tags": [
                        {"char_start": 1, "char_end": 11, "token": "2020-05-20", "tag": "PI"},
                        {"char_start": 12, "char_end": 20, "token": "John Doe", "tag": "PI"},
                        ]
                }
            ]
        verbose: e.g. False
    """
    os.makedirs("/".join(_output_file.split("/")[:-1]), exist_ok=True)
    if verbose:
        print(f"> write output_file = {_output_file}")
    with open(_output_file, "w") as f:
        for line in _output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_json(_input_file: str, verbose: bool = False) -> List[Dict[str, Any]]:
    if verbose:
        print(f"> read input_file = {_input_file}")
    with open(_input_file, "r") as f:
        input_lines = json.load(f)
    return input_lines


def write_json(_output_file: str, _output_lines: List[Dict[str, Any]], verbose: bool = False) -> None:
    if verbose:
        print(f"> write output_file = {_output_file}")
    with open(_output_file, "w") as f:
        f.write(json.dumps(_output_lines, ensure_ascii=False))
