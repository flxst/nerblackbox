import torch
from torch.utils.data import Dataset
from typing import List
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import InputExample
InputExamples = List[InputExample]


class BertDataset(Dataset):
    """ Bert Dataset with features created when requested """

    def __init__(self, input_examples: InputExamples, transform: callable) -> None:
        """
        Args:
            input_examples: [list] of [InputExample], e.g. text = 'at arbetsförmedlingen'
                                                           tags = '0 ORG'
            transform (callable): Transform to be applied on a list of InputExamples
        """
        self.input_examples = input_examples
        self.transform = transform

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """
        - apply self.transform on single sample w/ index = index

        Args:
            index: [int]

        Returns:
            input_ids:      [torch tensor], e.g. [1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]
            attention_mask: [torch tensor], e.g. [1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]
            segment_ids:    [torch tensor], e.g. [0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]
            tag_ids:        [torch tensor], e.g. [1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]
        """
        return self.transform(self.input_examples[index])
