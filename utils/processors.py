# coding=utf-8
# Copyright 2019 Arbetsförmedlingen AI-center.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, delimiter='\t', quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
class SentenceProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv"),  delimiter=";"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv", delimiter=";")), "dev")

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
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
   
class NERProcessor(DataProcessor):
    
    #wordpiece_conll_map = { 
    #    'O':'O', 'B_PER':'I_PER', 'B_ORG':'I_ORG', 'B_LOC':'I_LOC', 'B_MISC':'I_MISC',
    #    'I_PER':'I_PER', 'I_ORG':'I_ORG', 'I_LOC':'I_LOC', 'I_MISC':'I_MISC'
    #}
    
    #label_list = ['<pad>', '[CLS]','[SEP]', 'O', 
    #              'B_PER', 'B_ORG','B_LOC', 'B_MISC',
    #              'I_PER', 'I_ORG','I_LOC', 'I_MISC']
    
    wordpiece_conll_map = { 
        'O':'O', 'B_COMP':'I_COMP','I_COMP':'I_COMP'
    }
    
        
    label_list = ['<pad>', '[CLS]','[SEP]', 'O', 'B_COMP','I_COMP']
    
    def __init__(self, path, tokenizer, do_lower_case=True, separator='\t'):
        self.separator = separator
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

        data = pd.read_csv(path, names=['labels', 'text'], header=1, sep=self.separator)
        
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
    
class ConllNERProcessor(DataProcessor):
    
    wordpiece_conll_map = { 
        'O':'O', 'B_PER':'I_PER', 'B_ORG':'I_ORG', 'B_LOC':'I_LOC', 'B_MISC':'I_MISC',
        'I_PER':'I_PER', 'I_ORG':'I_ORG', 'I_LOC':'I_LOC', 'I_MISC':'I_MISC'
    }
    
    label_list = ['<pad>', '[CLS]','[SEP]', 'O', 
                  'B_PER', 'B_ORG','B_LOC', 'B_MISC',
                  'I_PER', 'I_ORG','I_LOC', 'I_MISC']
    
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
        data = pd.read_csv(path, names=['labels', 'text'], header=1)
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
    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            logger.debug("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.debug("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.debug("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

class BertDataset(TensorDataset):
    """ Bert Dataset. """

    def __init__(self, train_examples, tokenizer, max_seq_length=128, label_list=['0', '1'], transform=None):
        """
        Args:
            train_examples: a list of InputExample instances
            tokenizer: BertTokenizer used to tokenize to Wordpieces and transform to indices
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in self.train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in self.train_features], dtype=torch.long)
        
        self.tensors = (all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

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

