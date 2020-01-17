
import logging
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class InputExampleToTensors(object):
    """ Converts an InputExample to a tuple of feature tensors. """

    def __init__(self,
                 tokenizer,
                 max_seq_length: int = 128,
                 label_tuple: tuple = ('0', '1')):
        """
        :param tokenizer:      [BertTokenizer] used to tokenize to Wordpieces and transform to indices
        :param max_seq_length: [int]
        :param label_tuple:    [tuple] of [str]
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_tuple = label_tuple
        self.label2id = {label: i for i, label in enumerate(self.label_tuple)}

    def __call__(self, input_example):
        """
        transform input_example to tensors of length self.max_seq_length
        ----------------------------------
        :param input_example: [InputExample], e.g. text_a = 'at arbetsförmedlingen'
                                                   text_b = None
                                                   labels_a = '0 ORG'
                                                   labels_b = None
        :return: input_ids:   [torch tensor], e.g. [1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]
        :return: input_mask:  [torch tensor], e.g. [1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]
        :return: segment_ids: [torch tensor], e.g. [0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]
        :return: label_ids:   [torch tensor], e.g. [1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]  cf Processor
        """
        ####################
        # A0. tokens_*, token_labels_*
        ####################
        tokens_a, token_labels_a = self._tokenize_words_and_labels(input_example, segment='a')
        tokens_b, token_labels_b = self._tokenize_words_and_labels(input_example, segment='b')

        if tokens_b is None:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
        else:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)

        ####################
        # A1. tokens
        ####################
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]

        ####################
        # A2. label_ids
        ####################
        label_ids = [self.label2id['[CLS]']] + \
                    [self.label2id[token_label] for token_label in token_labels_a] + \
                    [self.label2id['[SEP]']]
        if tokens_b:
            label_ids += \
                [self.label2id[token_label] for token_label in token_labels_b] + \
                [self.label2id['[SEP]']]

        label_ids = self._pad_sequence(label_ids, 0)
        assert len(label_ids) == self.max_seq_length

        ####################
        # B. input_ids, input_mask, segment_ids
        ####################
        # 1. input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 2. input_mask
        input_mask = [1] * len(input_ids)  # 1 = real tokens, 0 = padding tokens. Only real tokens are attended to.

        # 3. segment_ids
        if tokens_b is None:
            segment_ids = [0] * len(tokens)
        else:
            segment_ids = [0] * len(tokens_a) + [1] * (len(tokens_b) + 1)

        # 4. padding
        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        ####################
        # return
        ####################
        return input_ids, input_mask, segment_ids, label_ids

    ####################################################################################################################
    # PRIVATE METHODS
    ####################################################################################################################
    def _tokenize_words_and_labels(self, input_example, segment):
        """
        gets NER labels for tokenized version of text
        ---------------------------------------------
        :param input_example: [InputExample], e.g. text_a = 'at arbetsförmedlingen'
                                                   text_b = None
                                                   labels_a = '0 ORG'
                                                   labels_b = None
        :param segment:       [str], 'a' or 'b'
        :changed attr: token_count [int] total number of tokens in df
        :return: tokens:        [list] of [str], e.g. ['at', 'arbetsförmedling', '##en]
                 tokens_labels: [list] of [str], e.g. [0, 'ORG', 'ORG']
        """
        # [list] of (word, label) pairs, e.g. [('at', '0'), ('Arbetsförmedlingen', 'ORG')]
        if segment == 'a':
            word_label_pairs = zip(input_example.text_a.split(' '), input_example.labels_a.split(' '))
        elif segment == 'b':
            if input_example.text_b is None or input_example.labels_b is None:
                return None, None
            else:
                word_label_pairs = zip(input_example.text_b.split(' '), input_example.labels_b.split(' '))
        else:
            raise Exception(f'> segment = {segment} unknown')

        tokens = []
        tokens_labels = []
        for word_label_pair in word_label_pairs:
            word, label = word_label_pair[0], word_label_pair[1]
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if label == 'O':
                b_label = label
                i_label = label
            else:
                b_label = label  # f'B-{label}'
                i_label = label  # f'I-{label}'
            tokens_labels.append(b_label)
            for _ in word_tokens[1:]:
                tokens_labels.append(i_label)

        return tokens, tokens_labels

    def _pad_sequence(self, _input, value):
        padded = pad_sequences(
            [_input],
            maxlen=self.max_seq_length,
            padding="post",
            value=value,
            dtype="long",
            truncating="post"
        )
        return torch.tensor(padded, dtype=torch.long).view(-1)

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
