import os
import pandas as pd
from .processors import logger
from .processors import InputExample
from .processors import DataProcessor


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SwedishNERCorpusProcessor(DataProcessor):
    wordpiece_conll_map = {
        'O': 'O', 'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC', 'MISC': 'MISC', 'ORG*': 'ORG',
    }

    label_list = ['<pad>', '[CLS]', '[SEP]', 'O',
                  'PER', 'ORG', 'LOC', 'MISC']

    def __init__(self, path, tokenizer, do_lower_case=True):
        self.train_data = self._read_csv(path + 'train.csv')
        self.valid_data = self._read_csv(path + 'valid.csv')
        self.test_data = self._read_csv(path + 'test.csv')
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case

    def get_train_examples(self):
        return self._create_examples(self.train_data, 'train')

    def get_val_examples(self):
        return self._create_examples(self.valid_data, 'val')

    def get_test_examples(self):
        return self._create_examples(self.test_data, 'test')

    def get_label_list(self):
        return self.label_list

    def _read_csv(self, path):
        data = pd.read_csv(path, names=['labels', 'text'], header=0, delimiter='\t')
        return data

    def _create_examples(self, data, set_type):
        examples = []
        self.token_count = 0

        for i, row in enumerate(data.itertuples()):
            if self.do_lower_case:
                text_a = row.text.lower()
            else:
                text_a = row.text

            labels = row.labels
            conll_sentence = zip(text_a.split(' '), labels.split(' '))

            guid = "%s-%s" % (set_type, i)
            labels = self.bert_labels(conll_sentence)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b='', label=labels))
        return examples

    def bert_labels(self, conll_sentence):
        bert_labels = []
        bert_labels.append('[CLS]')
        for conll in conll_sentence:
            self.token_count += 1
            token, label = conll[0], conll[1]
            bert_tokens = self.tokenizer.tokenize(token)
            bert_labels.append(label)
            for bert_token in bert_tokens[1:]:
                bert_labels.append(self.wordpiece_conll_map[label])
        return bert_labels


class SUCProcessor(DataProcessor):
    wordpiece_conll_map = {
        'O': 'O', 'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC', 'WRK': 'WRK',
        'OBJ': 'OBJ'
    }

    label_list = ['<pad>', '[CLS]', '[SEP]', 'O',
                  'PER', 'ORG', 'LOC', 'MISC', 'WRK', 'OBJ']

    def __init__(self, path, tokenizer, do_lower_case=True):
        self.train_data = self._read_csv(path + 'train.csv')
        self.valid_data = self._read_csv(path + 'valid.csv')
        self.test_data = self._read_csv(path + 'test.csv')
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case

    def get_train_examples(self):
        return self._create_examples(self.train_data, 'train')

    def get_val_examples(self):
        return self._create_examples(self.valid_data, 'val')

    def get_test_examples(self):
        return self._create_examples(self.test_data, 'test')

    def get_label_list(self):
        return self.label_list

    def _read_csv(self, path):
        data = pd.read_csv(path, names=['labels', 'text'], header=0, delimiter='\t')
        return data

    def _create_examples(self, data, set_type):
        examples = []
        self.token_count = 0

        for i, row in enumerate(data.itertuples()):
            if self.do_lower_case:
                text_a = row.text.lower()
            else:
                text_a = row.text

            labels = row.labels
            conll_sentence = zip(text_a.split(' '), labels.split(' '))

            guid = "%s-%s" % (set_type, i)
            labels = self.bert_labels(conll_sentence)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b='', label=labels))
        return examples

    def bert_labels(self, conll_sentence):
        bert_labels = ['[CLS]']
        for conll in conll_sentence:
            self.token_count += 1
            token, label = conll[0], conll[1]
            bert_tokens = self.tokenizer.tokenize(token)
            bert_labels.append(label)
            for _ in bert_tokens[1:]:
                bert_labels.append(self.wordpiece_conll_map[label])
        return bert_labels


class SUCIOBProcessor(DataProcessor):
    # wordpiece_conll_map = {
    #    'O':'O', 'B_PER':'I_PER', 'B_ORG':'I_ORG', 'B_LOC':'I_LOC', 'B_MISC':'I_MISC',
    #    'I_PER':'I_PER', 'I_ORG':'I_ORG', 'I_LOC':'I_LOC', 'I_MISC':'I_MISC'
    # }

    # label_list = ['<pad>', '[CLS]','[SEP]', 'O',
    #              'B_PER', 'B_ORG','B_LOC', 'B_MISC',
    #

    wordpiece_conll_map = {
        'O': 'O',
        'B_PER': 'B_PER',
        'B_ORG': 'B_ORG',
        'B_LOC': 'B_LOC',
        'B_TME': 'B_TME',
        'B_MSR': 'B_MSR',
        'B_EVN': 'B_EVN',
        'I_PER': 'I_PER',
        'I_ORG': 'I_ORG',
        'I_LOC': 'I_LOC',
        'I_TME': 'I_TME',
        'I_MSR': 'I_MSR',
        'I_EVN': 'I_EVN'
    }

    label_list = ['<pad>', '[CLS]', '[SEP]', 'O',
                  'B_PRS', 'I_PRS', 'B_ORG', 'I_ORG', 'I_LOC', 'B_LOC',
                  'B_TME', 'I_TME', 'B_MSR', 'I_MSR', 'B_EVN', 'I_EVN']

    def __init__(self, path, tokenizer, do_lower_case=True):
        self.train_data = self._read_csv(path + 'train.csv')
        self.valid_data = self._read_csv(path + 'valid.csv')
        self.test_data = self._read_csv(path + 'test.csv')
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case

    def get_train_examples(self):
        return self._create_examples(self.train_data, 'train')

    def get_val_examples(self):
        return self._create_examples(self.valid_data, 'val')

    def get_test_examples(self):
        return self._create_examples(self.test_data, 'test')

    def get_label_list(self):
        return self.label_list

    def _read_csv(self, path):
        data = pd.read_csv(path, names=['labels', 'text'], header=0, delimiter='\t')
        return data

    def _create_examples(self, data, set_type):
        examples = []
        self.token_count = 0

        for i, row in enumerate(data.itertuples()):
            if self.do_lower_case:
                text_a = row.text.lower()
            else:
                text_a = row.text

            labels = row.labels
            conll_sentence = zip(text_a.split(' '), labels.split(' '))

            guid = "%s-%s" % (set_type, i)
            labels = self.bert_labels(conll_sentence)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b='', label=labels))
        return examples

    def bert_labels(self, conll_sentence):
        bert_labels = []
        bert_labels.append('[CLS]')
        for conll in conll_sentence:
            self.token_count += 1
            token, label = conll[0], conll[1]
            bert_tokens = self.tokenizer.tokenize(token)
            bert_labels.append(label)
            for bert_token in bert_tokens[1:]:
                bert_labels.append(label)
        return bert_labels
