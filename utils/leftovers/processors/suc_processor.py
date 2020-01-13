
import pandas as pd
from utils.input_example import InputExample
from utils.leftovers.processors.data_processor import DataProcessor


########################################################################################################################
# from processors_additional.py
########################################################################################################################
class SUCProcessor(DataProcessor):

    wordpiece_conll_map = {
        'O': 'O',
        'PER': 'PER',
        'ORG': 'ORG',
        'LOC': 'LOC',
        'WRK': 'WRK',
        'OBJ': 'OBJ'
    }

    label_list = [
        '<pad>',
        '[CLS]',
        '[SEP]',
        'O',
        'PER',
        'ORG',
        'LOC',
        'MISC',
        'WRK',
        'OBJ'
    ]

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
            for _ in bert_tokens[1:]:
                bert_labels.append(self.wordpiece_conll_map[label])
        return bert_labels
