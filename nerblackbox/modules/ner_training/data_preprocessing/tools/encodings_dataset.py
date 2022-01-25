import torch
from torch.utils.data import Dataset
from typing import Dict
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    Encodings,
)


class EncodingsDataset(Dataset):
    """Encodings Dataset with features created when requested"""

    def __init__(self, encodings: Encodings) -> None:
        """
        Args:
            encodings: [Encodings] with
            key = input_ids,      value = [2D torch tensor], e.g. [[1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]]
            key = attention_mask, value = [2D torch tensor], e.g. [[1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]]
            key = token_type_ids, value = [2D torch tensor], e.g. [[0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]]
            key = labels,         value = [2D torch tensor], e.g. [[1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]]
        """
        self.encodings = encodings

    def __len__(self) -> int:
        return self.encodings["labels"].shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            index: [int]

        Returns:
            sample = Dict with keys = subset of EncodingsKeys, values = 2D torch tensor

            e.g.
            input_ids:      [2D torch tensor], e.g. [[1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]]
            attention_mask: [2D torch tensor], e.g. [[1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]]
            token_type_ids: [2D torch tensor], e.g. [[0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]]
            labels:         [2D torch tensor], e.g. [[1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]]
        """
        return {
            key: self.encodings[key][index]
            for key in self.encodings
            if len(self.encodings[key])
        }
