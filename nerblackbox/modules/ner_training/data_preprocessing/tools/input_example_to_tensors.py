import torch
import numpy as np
from typing import List
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import InputExample


class InputExampleToTensors:
    """
    Converts an InputExample to a tuple of feature tensors:
    (input_ids, attention_mask, segment_ids, tag_ids)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 128,
        tag_tuple: tuple = ("O", "PER", "ORG"),
        default_logger=None,
    ):
        """
        :param tokenizer:      [BertTokenizer] used to tokenize to Wordpieces and transform to indices
        :param max_seq_length: [int]
        :param tag_tuple:      [tuple] of [str]
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.default_logger = default_logger

        self.tag2id = {tag: i for i, tag in enumerate(tag_tuple)}
        if self.default_logger:
            self.default_logger.log_debug("> tag2id:", self.tag2id)

    def __call__(self, input_example: InputExample) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """
        transform input_example to tensors of length self.max_seq_length
        ----------------------------------------------------------------
        :param input_example: [InputExample], e.g. text = 'at arbetsförmedlingen'
                                                   tags = '0 ORG'
        :return: input_ids:      [torch tensor], e.g. [1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]
        :return: attention_mask: [torch tensor], e.g. [1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]
        :return: segment_ids:    [torch tensor], e.g. [0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]
        :return: tag_ids:        [torch tensor], e.g. [1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]
        """
        ####################
        # A0. tokens_*, tags_*
        ####################
        tokens_split_into_words = input_example.text.split()
        tags_split_into_words = input_example.tags.split()
        assert len(tokens_split_into_words) == len(tags_split_into_words), \
            f"ERROR! len(tokens) = {len(tokens_split_into_words)} is different from len(tags) = {len(tags_split_into_words)}"

        encodings = self.tokenizer(
            tokens_split_into_words,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            is_split_into_words=True,
            return_offsets_mapping=True,
        )
        input_ids = torch.tensor(encodings["input_ids"])
        attention_mask = torch.tensor(encodings["attention_mask"])
        segment_ids = torch.tensor(encodings["token_type_ids"])
        tag_ids = torch.tensor(self._encode_tags(tags_split_into_words, encodings.offset_mapping))

        return input_ids, attention_mask, segment_ids, tag_ids

    ####################################################################################################################
    # PRIVATE HELPER METHODS
    ####################################################################################################################
    def _encode_tags(self, _tags_split_into_words: List[str], offsets: List[List[int]]) -> List[int]:
        tag_ids_split_into_words: List[int] = [self.tag2id[tag] for tag in _tags_split_into_words]

        # create an empty array of -100
        arr_tag_ids: np.array = np.ones(len(offsets), dtype=int) * -100
        arr_offsets: np.array = np.array(offsets)

        # set labels whose first offset position is 0 and the second is not 0
        nr_matches: int = len([elem for elem in arr_offsets if elem[0] == 0 and elem[1] != 0])
        arr_tag_ids[(arr_offsets[:, 0] == 0) & (arr_offsets[:, 1] != 0)] = tag_ids_split_into_words[:nr_matches]

        # convert
        tag_ids: List[int] = arr_tag_ids.tolist()

        return tag_ids
