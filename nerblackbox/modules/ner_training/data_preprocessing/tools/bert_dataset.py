import torch
from torch.utils.data import Dataset
from typing import Tuple
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    Encodings,
    EncodingsKeys,
)


class BertDataset(Dataset):
    """ Bert Dataset with features created when requested """

    def __init__(self, encodings: Encodings) -> None:
        """
        Args:
            encodings: [Encodings] with
            key = input_ids,      value = [2D torch tensor], e.g. [[1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]]
            key = attention_mask, value = [2D torch tensor], e.g. [[1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]]
            key = token_type_ids, value = [2D torch tensor], e.g. [[0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]]
            key = tag_ids,        value = [2D torch tensor], e.g. [[1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]]
        """
        self.encodings = encodings

    def __len__(self) -> int:
        return self.encodings["tag_ids"].shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            index: [int]

        Returns:
            input_ids:      [2D torch tensor], e.g. [[1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]]
            attention_mask: [2D torch tensor], e.g. [[1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]]
            token_type_ids: [2D torch tensor], e.g. [[0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]]
            tag_ids:        [2D torch tensor], e.g. [[1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]]
        """
        return tuple([self.encodings[key][index] for key in EncodingsKeys])
