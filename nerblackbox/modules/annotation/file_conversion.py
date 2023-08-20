from typing import Optional, List, Dict, Any


########################################################################################################################
# nerblackbox <-> labelstudio
########################################################################################################################
def nerblackbox2labelstudio(
    _input_lines: List[Dict[str, Any]], _max_lines: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    convert data from nerblackbox to labelstudio format

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
        _max_lines: e.g. 30

    Returns:
        _output_lines: e.g.
            [
                {
                    {
                        "data": {
                            "text": "\n2020-05-20 John Doe pratar.",
                        },
                        "annotations": [{
                            "result": [
                                {
                                    "id": "0",
                                    [..]
                                    "value": {
                                        "start": 1,
                                        "end": 11,
                                        "text": "2020-05-20",
                                        "labels": ["PI"],
                                    },
                                },
                                {
                                    "id": "1",
                                    [..]
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
                }
            ]
    """
    if _max_lines is not None:
        _input_lines = _input_lines[:_max_lines]

    _output_lines = list()
    idx = 0
    for i, input_line in enumerate(_input_lines):
        output_line = {
            "data": {
                "text": input_line["text"],
            },
            "annotations": [
                {
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
                }
            ],
        }
        idx += len(input_line["tags"])
        _output_lines.append(output_line)

    return _output_lines


def labelstudio2nerblackbox(
    _input_lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    convert data from labelstudio to nerblackbox format

    Args:
        _input_lines: e.g.
            [
                {
                    {
                        "data": {
                            "text": "\n2020-05-20 John Doe pratar.",
                        },
                        "annotations": [{
                            "result": [
                                {
                                    "id": "0",
                                    [..]
                                    "value": {
                                        "start": 1,
                                        "end": 11,
                                        "text": "2020-05-20",
                                        "labels": ["PI"],
                                    },
                                },
                                {
                                    "id": "1",
                                    [..]
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
                }
            ]

    Returns:
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
    """
    _output_lines = list()
    for input_line in _input_lines:
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
                if label["type"] == "labels"
            ],
        }
        _output_lines.append(output_line)

    return _output_lines


########################################################################################################################
# nerblackbox <-> doccano
########################################################################################################################
def nerblackbox2doccano(
    _input_lines: List[Dict[str, Any]], _max_lines: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    convert data from nerblackbox to doccano format

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
        _max_lines: e.g. 30

    Returns:
        _output_lines: e.g.
            [
                {
                    "text": "\n2020-05-20 John Doe pratar.",
                    "label": [
                        [1, 11, "PI"],
                        [12, 20, "PI"],
                    ]
                }
            ]
    """
    if _max_lines is not None:
        _input_lines = _input_lines[:_max_lines]

    _output_lines = list()
    for input_line in _input_lines:
        output_line = {
            "text": input_line["text"],
            "label": [
                [int(tag["char_start"]), int(tag["char_end"]), tag["tag"]]
                for tag in input_line["tags"]
            ],
        }
        _output_lines.append(output_line)

    return _output_lines


def doccano2nerblackbox(_input_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    convert data from doccano to nerblackbox format

    Args:
        _input_lines: e.g.
            [
                {
                    "text": "\n2020-05-20 John Doe pratar.",
                    "label": [
                        [1, 11, "PI"],
                        [12, 20, "PI"],
                    ]
                }
            ]

    Returns:
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
    """
    _output_lines = list()
    for input_line in _input_lines:
        output_line = {
            "text": input_line["text"],
            "tags": [
                {
                    "char_start": label[0],
                    "char_end": label[1],
                    "token": input_line["text"][label[0] : label[1]],
                    "tag": label[2],
                }
                for label in input_line["label"]
            ],
        }
        _output_lines.append(output_line)

    return _output_lines
