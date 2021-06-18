import torch
from typing import Dict, List
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)

Encodings = Dict[str, torch.Tensor]
EncodingsKeys = ["input_ids", "attention_mask", "token_type_ids", "tag_ids"]
InputExamples = List[InputExample]
