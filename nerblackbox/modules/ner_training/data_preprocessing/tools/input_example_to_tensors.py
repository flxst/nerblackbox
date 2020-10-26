import torch
from torch.nn.utils.rnn import pad_sequence


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

    def __call__(self, input_example):
        """
        transform input_example to tensors of length self.max_seq_length
        ----------------------------------------------------------------
        :param input_example: [InputExample], e.g. text_a = 'at arbetsförmedlingen'
                                                   text_b = None
                                                   tags_a = '0 ORG'
                                                   tags_b = None
        :return: input_ids:      [torch tensor], e.g. [1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]
        :return: attention_mask: [torch tensor], e.g. [1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]
        :return: segment_ids:    [torch tensor], e.g. [0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]
        :return: tag_ids:        [torch tensor], e.g. [1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]
        """
        ####################
        # A0. tokens_*, tags_*
        ####################
        tokens_a, tags_a = self._tokenize_words_and_tags(input_example, segment="a")
        tokens_b, tags_b = self._tokenize_words_and_tags(input_example, segment="b")

        # Modify `tokens_a` (and `tokens_b`) in place so that the total length is less than the specified length.
        if tokens_b is None:
            # Account for [CLS] and [SEP] with "- 2"
            self._truncate_seq_pair(self.max_seq_length - 2, tokens_a)
            self._truncate_seq_pair(self.max_seq_length - 2, tags_a)
        else:
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(self.max_seq_length - 3, tokens_a, tokens_b)
            self._truncate_seq_pair(self.max_seq_length - 3, tags_a, tags_b)

        ####################
        # A1. tokens, tags
        ####################
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tags = ["[CLS]"] + tags_a + ["[SEP]"]
        if tokens_b and tags_b:
            tokens += tokens_b + ["[SEP]"]
            tags += tags_b + ["[SEP]"]

        ####################
        # B. input_ids, attention_mask, segment_ids, tag_ids
        ####################
        # 1. input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 2. attention_mask
        attention_mask = [1] * len(input_ids)  # 1 = real tokens, 0 = padding tokens.

        # 3. segment_ids
        if tokens_b is None:
            segment_ids = [0] * len(tokens)
        else:
            segment_ids = [0] * len(tokens_a) + [1] * (len(tokens_b) + 1)

        # 4. tag_ids
        tag_ids = [self.tag2id[tag] for tag in tags]

        # 5. cast to tensor & padding
        input_ids = self._pad_sequence(input_ids, 0)
        attention_mask = self._pad_sequence(attention_mask, 0)
        segment_ids = self._pad_sequence(segment_ids, 0)
        tag_ids = self._pad_sequence(tag_ids, 0)
        assert (
            input_ids.shape[0] == self.max_seq_length
        ), f"shape[0] = {input_ids[0].shape}"
        assert (
            attention_mask.shape[0] == self.max_seq_length
        ), f"shape[0] = {attention_mask[0].shape}"
        assert (
            segment_ids.shape[0] == self.max_seq_length
        ), f"shape[0] = {segment_ids[0].shape}"
        assert tag_ids.shape[0] == self.max_seq_length, f"shape[0] = {tag_ids[0].shape}"

        ####################
        # return
        ####################
        return input_ids, attention_mask, segment_ids, tag_ids

    ####################################################################################################################
    # PRIVATE HELPER METHODS
    ####################################################################################################################
    def _tokenize_words_and_tags(self, input_example, segment):
        """
        gets NER tags for tokenized version of text
        ---------------------------------------------
        :param input_example: [InputExample], e.g. text_a = 'at arbetsförmedlingen'
                                                   text_b = None
                                                   tags_a = '0 ORG'
                                                   tags_b = None
        :param segment:       [str], 'a' or 'b'
        :changed attr: token_count [int] total number of tokens in df
        :return: tokens: [list] of [str], e.g. ['at', 'arbetsförmedling', '##en]
                 tags:   [list] of [str], e.g. [   0,              'ORG', 'ORG']
        """
        # [list] of (word, tag) pairs, e.g. [('at', '0'), ('Arbetsförmedlingen', 'ORG')]
        if segment == "a":
            word_tag_pairs = zip(
                input_example.text_a.split(" "), input_example.tags_a.split(" ")
            )
        elif segment == "b":
            if input_example.text_b is None or input_example.tags_b is None:
                return None, None
            else:
                word_tag_pairs = zip(
                    input_example.text_b.split(" "), input_example.tags_b.split(" ")
                )
        else:
            raise Exception(f"> segment = {segment} unknown")

        tokens = []
        tokens_tags = []
        for word_tag_pair in word_tag_pairs:
            word, tag = word_tag_pair[0], word_tag_pair[1]
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            tokens_tags.append(tag)
            for _ in word_tokens[1:]:
                tokens_tags.append(
                    tag.replace("B-", "I-")
                )  # replace only applies to IOB tags

        return tokens, tokens_tags

    @staticmethod
    def _truncate_seq_pair(max_length, seq_a, seq_b=()):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(seq_a) + len(seq_b)
            if total_length <= max_length:
                break
            if len(seq_a) > len(seq_b):
                seq_a.pop()
            else:
                seq_b.pop()

    def _pad_sequence(self, _input_list, padding_value):
        """
        pad _input sequence with value until self.max_seq_length is reached
        -------------------------------------------------------------------
        :param _input_list:   [list] of [int]
        :param padding_value: [int], e.g. 0
        :return: padded _input as [torch tensor]
        """
        padded = pad_sequence(
            [torch.tensor(_input_list), torch.zeros(self.max_seq_length)],
            batch_first=True,
            padding_value=padding_value,
        )
        return padded[0]
