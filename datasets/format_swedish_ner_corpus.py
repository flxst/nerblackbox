
import os
import json

import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import get_dataset_path


def read_txt(phase):
    """
    - read original swedish_ner_corpus text files
    ---------------------------------------------
    :param phase:   [str] 'train' or 'test'
    :return: _rows: [list] of [list] of [str], e.g. [['Inger', 'PER'], ['säger', '0'], ..]
    """
    file_path = f'datasets/ner/swedish_ner_corpus/{phase}_corpus.txt'

    _rows = list()
    with open(file_path) as f:
        for row in f.readlines():
            _rows.append(row.strip().split())

    print(f'> read {file_path}')

    return _rows


def write_csv(phase, rows, dataset_path):
    """
    - write reshaped swedish_ner_corpus text files
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

        labels = list()
        sentence = list()
        for row in rows:
            if len(row) == 2:
                sentence.append(row[0])
                labels.append(ner_label_mapping[row[1]] if row[1] != '0' else 'O')  # replace zeros by capital O (!)
                if row[0] == '.':
                    f.write(' '.join(labels) + '\t' + ' '.join(sentence) + '\n')
                    labels = list()
                    sentence = list()

    print(f'> wrote {file_path}')


def main():
    """
    - read original swedish_ner_corpus text files
    - split valid/test
    - reshape text files to standard format
    --------------------------------------------------------------------------------
    :return: -
    """
    # train
    rows = read_txt('train')
    dataset_path = os.path.join(BASE_DIR, get_dataset_path('swedish_ner_corpus'))
    write_csv('train', rows, dataset_path=dataset_path)

    # valid/test
    rows = read_txt('test')
    split_index = int(len(rows)/2.)
    rows_valid = rows[:split_index]
    rows_test = rows[split_index:]
    write_csv('valid', rows_valid, dataset_path=dataset_path)
    write_csv('test', rows_test, dataset_path=dataset_path)


if __name__ == '__main__':
    main()
