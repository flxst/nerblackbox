import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    Encodings,
    EncodingsKeys,
)


class InputExamplesToTensors:
    """
    Converts List[InputExample] to Encodings w/ keys = "input_ids", "attention_mask", "token_type_ids", "tag_ids"
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

    def __call__(self, input_examples: List[InputExample], predict: bool) -> Encodings:
        """
        Args:
            input_examples: List[InputExample] with e.g. text = 'at arbetsförmedlingen'
                                                         tags = '0 ORG'
            predict: if True,  map special tokens to    O tag_id for prediction (will not be used)
                     if False, map special tokens to -100 tag_id for train, val, test

        Returns:
            encodings: [Encodings] with
            key = input_ids,      value = [2D torch tensor], e.g. [[1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]]
            key = attention_mask, value = [2D torch tensor], e.g. [[1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]]
            key = token_type_ids, value = [2D torch tensor], e.g. [[0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]]
            key = tag_ids,        value = [2D torch tensor], e.g. [[1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]]
        """

        # encodings_single
        encodings_single: Dict[str, List[Any]] = {key: list() for key in EncodingsKeys}
        for input_example in input_examples:
            _encodings = self._transform_input_example(input_example, predict)
            for key in encodings_single.keys():
                encodings_single[key].append(_encodings[key])

        # combine encodings_single -> encodings
        encodings = {
            key: torch.cat(encodings_single[key]) for key in encodings_single.keys()
        }

        return encodings

    def _transform_input_example(
        self, input_example: InputExample, predict: bool
    ) -> Encodings:
        """
        - transform input_example to tensors of length self.max_seq_length

        Args:
            input_example: [InputExample], with e.g. text = 'at arbetsförmedlingen'
                                                     tags = '0 ORG'
            predict: if True,  map special tokens to    O tag_id for prediction (will not be used)
                     if False, map special tokens to -100 tag_id for train, val, test

        Returns:
            encodings: [Encodings] with
            key = input_ids,      value = [2D torch tensor], e.g. [[1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]]
            key = attention_mask, value = [2D torch tensor], e.g. [[1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]]
            key = token_type_ids, value = [2D torch tensor], e.g. [[0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]]
            key = tag_ids,        value = [2D torch tensor], e.g. [[1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]]
        """
        ####################
        # A0. tokens_*, tags_*
        ####################
        tokens_split_into_words = input_example.text.split()
        tags_split_into_words = input_example.tags.split()
        assert len(tokens_split_into_words) == len(
            tags_split_into_words
        ), f"ERROR! len(tokens) = {len(tokens_split_into_words)} is different from len(tags) = {len(tags_split_into_words)}"

        encodings = self.tokenizer(
            tokens_split_into_words,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            is_split_into_words=True,
            return_offsets_mapping=True,
            stride=0,
            return_overflowing_tokens=True,
        )

        encodings["tag_ids"] = self._encode_tags(
            tags_split_into_words, encodings.offset_mapping, predict
        )
        keys = list(encodings.keys())
        for key in keys:
            if key in EncodingsKeys:
                encodings[key] = torch.tensor(encodings[key])
            else:
                encodings.pop(key)

        return encodings

    ####################################################################################################################
    # PRIVATE HELPER METHODS
    ####################################################################################################################
    def _encode_tags(
        self,
        _tags_split_into_words: List[str],
        all_offsets: List[List[Tuple[int, int]]],
        predict: bool,
    ) -> List[List[int]]:
        """
        Args:
            _tags_split_into_words: ['at arbetsförmedlingen']
            all_offsets: [chunks] with chunk = [(start_1, end_1), (start_2, end_2), ..], e.g. [[(0, 2), (2, 5), ..]
            predict: if True,  map special tokens to    O tag_id for prediction (will not be used)
                     if False, map special tokens to -100 tag_id for train, val, test

        Returns:
            all_tag_ids: [chunk_tag_ids] with chunk_tag_ids = e.g. [-100, 3, -100, 4, 5, -100, ..]
        """

        tag_ids_split_into_words: List[int] = [
            self.tag2id[tag] for tag in _tags_split_into_words
        ]

        tag_id_special = 0 if predict else -100

        index = 0
        all_tag_ids = list()
        for offsets in all_offsets:
            # create an empty array of -100
            arr_tag_ids: np.array = np.ones(len(offsets), dtype=int) * tag_id_special
            arr_offsets: np.array = np.array(offsets)

            # set labels whose first offset position is 0 and the second is not 0
            nr_matches: int = len(
                [elem for elem in arr_offsets if elem[0] == 0 and elem[1] != 0]
            )
            arr_tag_ids[
                (arr_offsets[:, 0] == 0) & (arr_offsets[:, 1] != 0)
            ] = tag_ids_split_into_words[index : index + nr_matches]
            index += nr_matches

            # convert
            tag_ids: List[int] = arr_tag_ids.tolist()
            all_tag_ids.append(tag_ids)

        return all_tag_ids
