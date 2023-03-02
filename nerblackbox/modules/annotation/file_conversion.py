import os
import json
from typing import Optional


def nerblackbox2labelstudio(
        _input_file: str, _output_file: str, _max_lines: Optional[int] = None, verbose: bool = False
):
    """
    convert data from nerblackbox to labelstudio format

    Args:
        _input_file: e.g. '[..]/batch_1.jsonl
        _output_file: e.g. '[..]/batch_1_LABELSTUDIO.json'
        _max_lines: e.g. 30
        verbose:
    """
    if verbose:
        print(f"> read input_file = {_input_file}")
    with open(_input_file, "r") as f:
        input_lines = [json.loads(line) for line in f]

    if _max_lines is not None:
        input_lines = input_lines[:_max_lines]

    output_lines = list()
    idx = 0
    for i, input_line in enumerate(input_lines):
        output_line = {
            "data": {
                "text": input_line["text"],
            },
            "annotations": [{
                "result": [
                    {
                        "id": str(idx + j),
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": tag["char_start"],
                            "end": tag["char_end"],
                            "text": tag["token"],
                            "labels": [tag["tag"]],
                        },
                    }
                    for j, tag in enumerate(input_line["tags"])
                ],
            }],
        }
        idx += len(input_line["tags"])
        output_lines.append(output_line)

    if verbose:
        print(f"> write output_file = {_output_file}")
    with open(_output_file, "w") as f:
        f.write(json.dumps(output_lines, ensure_ascii=False))


def doccano2nerblackbox(_input_file: str, _output_file: str, verbose: bool = False) -> None:
    """
    convert data from doccano to nerblackbox format

    Args:
        _input_file: e.g. '[..]/batch_1_DOCCANO.jsonl'
        _output_file: e.g. '[..]/batch_1.jsonl
        verbose:
    """
    if verbose:
        print(f"> read input_file = {_input_file}")
    with open(_input_file, "r") as f:
        input_lines = [json.loads(line) for line in f]

    output_lines = list()
    for input_line in input_lines:
        output_line = {
            "text": input_line["text"],
            "tags": [
                {
                    "char_start": label[0],
                    "char_end": label[1],
                    "token": input_line["text"][label[0]: label[1]],
                    "tag": label[2],
                }
                for label in input_line["label"]
            ],
        }
        output_lines.append(output_line)

    if verbose:
        print(f"> write output_file = {_output_file}")

    os.makedirs("/".join(_output_file.split("/")[:-1]), exist_ok=True)
    with open(_output_file, "w") as f:
        for line in output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def nerblackbox2doccano(
    _input_file: str, _output_file: str, _max_lines: Optional[int] = None, verbose: bool = False
) -> None:
    """
    convert data from nerblackbox to doccano format

    Args:
        _input_file: e.g. '[..]/batch_1.jsonl
        _output_file: e.g. '[..]/batch_1_DOCCANO.jsonl'
        _max_lines: e.g. 30
        verbose:
    """

    if verbose:
        print(f"> read input_file = {_input_file}")
    with open(_input_file, "r") as f:
        input_lines = [json.loads(line) for line in f]

    if _max_lines is not None:
        input_lines = input_lines[:_max_lines]

    output_lines = list()
    for input_line in input_lines:
        output_line = {
            "text": input_line["text"],
            "label": [
                [int(tag["char_start"]), int(tag["char_end"]), tag["tag"]]
                for tag in input_line["tags"]
            ],
        }
        output_lines.append(output_line)

    if verbose:
        print(f"> write output_file = {_output_file}")
    with open(_output_file, "w") as f:
        for line in output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def labelstudio2nerblackbox(_input_file: str, _output_file: str, verbose: bool = False):
    """
    convert data from labelstudio to nerblackbox format

    Args:
        _input_file: e.g. '[..]/batch_1_LABELSTUDIO.json'
        _output_file: e.g. '[..]/batch_1.jsonl
        verbose:
    """
    if verbose:
        print(f"> read input_file = {_input_file}")
    with open(_input_file, "r") as f:
        input_lines = json.load(f)

    output_lines = list()
    for input_line in input_lines:
        output_line = {
            "text": input_line["data"]["text"],
            "tags": [
                {
                    "char_start": label["value"]["start"],
                    "char_end": label["value"]["end"],
                    "token": label["value"]["text"],
                    "tag": label["value"]["labels"][0],
                }
                for label in input_line["annotations"][0]["result"]
            ]
        }
        output_lines.append(output_line)

    if verbose:
        print(f"> write output_file = {_output_file}")

    os.makedirs("/".join(_output_file.split("/")[:-1]), exist_ok=True)
    with open(_output_file, "w") as f:
        for line in output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
