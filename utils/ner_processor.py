
import pandas as pd
import json
import logging
from utils.input_example import InputExample

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NerProcessor:

    def __init__(self, path, tokenizer, do_lower_case=True, separator='\t'):
        self.separator = separator
        with open(path + 'wordpiece_conll_map.json', 'r') as f:
            self.wordpiece_conll_map = json.load(f)
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
        return ['<pad>', '[CLS]', '[SEP]'] + [key.replace('*', '') for key in self.wordpiece_conll_map.keys()]

    def _read_csv(self, path):
        data = pd.read_csv(path, names=['labels', 'text'], header=0, sep=self.separator)
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
            labels = self._bert_labels(conll_sentence)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b='', label=labels))
        return examples

    def _bert_labels(self, conll_sentence):
        bert_labels = ['[CLS]']
        for conll in conll_sentence:
            self.token_count += 1
            token, label = conll[0], conll[1]
            bert_tokens = self.tokenizer.tokenize(token)
            bert_labels.append(label)
            for _ in bert_tokens[1:]:
                bert_labels.append(self.wordpiece_conll_map[label])
        return bert_labels
