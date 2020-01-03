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

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

class BertDataset(Dataset):
    """ Bert Dataset with features created when requested """

    def __init__(self, samples, transform, max_len=None):
        """
        Args
            samples: List of InputExample instances.
            transform (callable): Transform to be applied on a list of InputExamples
        """
        self.samples = samples
        self.transform = transform
        self.max_len = max_len
        
    def __len__(self):
        if self.max_len: return self.max_len
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.transform(self.samples[index])
    

class InputExampleToTensors(object):
    """ Converts a InputExample to a tuple of feature tensors.

    Args:
        train_examples: a list of InputExample instances
        tokenizer: BertTokenizer used to tokenize to Wordpieces and transform to indices
    """

    def __init__(self, tokenizer, max_seq_length=128, label_list=['0', '1']):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_list = label_list

    def __call__(self, example):
        
        label_map = {label : i for i, label in enumerate(self.label_list)}
        
        tokens_a = self.tokenizer.tokenize(example.text_a)
        
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
                
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

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        
        if isinstance(example.label, list):
            label_id = [label_map[label] for label in example.label]
            #label_padding = [0] * (self.max_seq_length - len(label_id))
            #label_id += label_padding
            #label_id = torch.tensor(label_id, dtype=torch.long)
            
            label_id = self._pad_sequence(label_id, self.max_seq_length, 0)
            assert len(label_id) == self.max_seq_length
        else:
            label_id = label_map[example.label]
            label_id = torch.tensor(label_id, dtype=torch.long)
            
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
           
        return (input_ids, input_mask, segment_ids, label_id)
    
    def _pad_sequence(self, input, maxlen, value):
        padded = pad_sequences([input], maxlen=self.max_seq_length, padding="post", value=value, dtype="long", truncating="post")
        return torch.tensor(padded, dtype=torch.long).view(-1)
     
    
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
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
