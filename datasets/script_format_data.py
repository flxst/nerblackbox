
import argparse

import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import get_dataset_path
from datasets.formatter.swedish_ner_corpus_formatter import SwedishNerCorpusFormatter


def main(args):
    """
    - read original text files
    - split valid/test
    - reshape text files to standard format
    --------------------------------------------------------------------------------
    :return: -
    """
    if args.ner_dataset == 'swedish_ner_corpus':
        # formatter
        formatter = SwedishNerCorpusFormatter()

        # train
        rows = formatter.read_original_file('train')
        dataset_path = os.path.join(BASE_DIR, get_dataset_path('swedish_ner_corpus'))
        formatter.write_formatted_csv('train', rows, dataset_path=dataset_path)

        # valid/test
        rows = formatter.read_original_file('test')
        split_index = int(len(rows)*args.valid_fraction)
        rows_valid = rows[:split_index]
        rows_test = rows[split_index:]
        formatter.write_formatted_csv('valid', rows_valid, dataset_path=dataset_path)
        formatter.write_formatted_csv('test', rows_test, dataset_path=dataset_path)
    else:
        raise Exception(f'ner_dataset = {args.ner_dataset} unknown.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--valid_fraction', type=float, default=1.0)
    _args = parser.parse_args()

    main(_args)
