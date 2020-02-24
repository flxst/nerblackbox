
import json

import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.base_formatter import BaseFormatter


class SwedishNerCorpusFormatter(BaseFormatter):

    def __init__(self):
        ner_dataset = 'swedish_ner_corpus'
        ner_label_list = ['PER', 'ORG', 'LOC', 'MISC', 'PRG', 'ORG*']
        super().__init__(ner_dataset, ner_label_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def modify_ner_label_mapping(self, ner_label_mapping_original, with_tags: bool):
        """
        customize ner label mapping if wanted
        -------------------------------------
        :param ner_label_mapping_original: [dict] w/ keys = labels in original data, values = labels in original data
        :param with_tags: [bool], if True: create labels with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple labels, e.g. 'PER', 'LOC', ..
        :return: ner_label_mapping: [dict] w/ keys = labels in original data, values = labels in formatted data
        """
        ner_label_mapping = ner_label_mapping_original

        # take care of extra case: ORG*
        if with_tags:
            ner_label_mapping['B-ORG*'] = 'B-ORG'
            ner_label_mapping['I-ORG*'] = 'I-ORG'
            ner_label_mapping['B-PRG'] = 'O'
            ner_label_mapping['I-PRG'] = 'O'
        else:
            ner_label_mapping['ORG*'] = 'ORG'
            ner_label_mapping['PRG'] = 'O'

        return ner_label_mapping

    def read_original_file(self, phase):
        """
        - read original text files
        ---------------------------------------------
        :param phase:   [str] 'train' or 'test'
        :return: _rows: [list] of [list] of [str], e.g. [['Inger', 'PER'], ['säger', '0'], ..]
        """
        file_path = f'datasets/ner/swedish_ner_corpus/{phase}_corpus.txt'

        _rows = list()
        with open(file_path) as f:
            for row in f.readlines():
                _rows.append(row.strip().split())

        print()
        print(f'> read {file_path}')

        return _rows

    def write_formatted_csv(self, phase, rows, dataset_path):
        """
        - write formatted csv files
        ----------------------------------------------
        :param phase:         [str] 'train' or 'test'
        :param rows:          [list] of [list] of [str], e.g. [['Inger', 'PER'], ['säger', '0'], ..]
        :param dataset_path:  [str] relative path from BASE_DIR to swedish_ner_corpus directory
        :return: -
        """
        file_path = f'datasets/ner/swedish_ner_corpus/{phase}.csv'

        # ner label mapping
        with open(os.path.join(dataset_path, 'ner_label_mapping.json'), 'r') as f:
            ner_label_mapping = json.load(f)

        # processing
        with open(file_path, mode='w') as f:

            num_sentences = 0
            labels = list()
            sentence = list()
            for row in rows:
                if len(row) == 2:
                    sentence.append(row[0])
                    labels.append(ner_label_mapping[row[1]] if row[1] != '0' else 'O')  # replace zeros by capital O (!)
                    if row[0] == '.':
                        f.write(' '.join(labels) + '\t' + ' '.join(sentence) + '\n')
                        num_sentences += 1
                        labels = list()
                        sentence = list()

        print(f'> phase = {phase}: wrote {len(rows)} words in {num_sentences} sentences to {file_path}')
