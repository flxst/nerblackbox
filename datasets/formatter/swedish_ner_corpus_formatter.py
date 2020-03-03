
import json

from os.path import abspath, dirname, join
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.base_formatter import BaseFormatter


class SwedishNerCorpusFormatter(BaseFormatter):

    def __init__(self):
        ner_dataset = 'swedish_ner_corpus'
        ner_tag_list = ['PER', 'ORG', 'LOC', 'MISC', 'PRG', 'ORG*']
        super().__init__(ner_dataset, ner_tag_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def modify_ner_tag_mapping(self, ner_tag_mapping_original, with_tags: bool):
        """
        customize ner tag mapping if wanted
        -------------------------------------
        :param ner_tag_mapping_original: [dict] w/ keys = tags in original data, values = tags in original data
        :param with_tags: [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple tags, e.g. 'PER', 'LOC', ..
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        ner_tag_mapping = ner_tag_mapping_original

        # take care of extra case: ORG*
        if with_tags:
            ner_tag_mapping['B-ORG*'] = 'B-ORG'
            ner_tag_mapping['I-ORG*'] = 'I-ORG'
            ner_tag_mapping['B-PRG'] = 'O'
            ner_tag_mapping['I-PRG'] = 'O'
        else:
            ner_tag_mapping['ORG*'] = 'ORG'
            ner_tag_mapping['PRG'] = 'O'

        return ner_tag_mapping

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

        # ner tag mapping
        with open(join(dataset_path, 'ner_tag_mapping.json'), 'r') as f:
            ner_tag_mapping = json.load(f)

        # processing
        with open(file_path, mode='w') as f:

            num_sentences = 0
            tags = list()
            sentence = list()
            for row in rows:
                if len(row) == 2:
                    sentence.append(row[0])
                    tags.append(ner_tag_mapping[row[1]] if row[1] != '0' else 'O')  # replace zeros by capital O (!)
                    if row[0] == '.':
                        f.write(' '.join(tags) + '\t' + ' '.join(sentence) + '\n')
                        num_sentences += 1
                        tags = list()
                        sentence = list()

        print(f'> phase = {phase}: wrote {len(rows)} words in {num_sentences} sentences to {file_path}')
