
import os
import json
import argparse

import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.swedish_ner_corpus_formatter import SwedishNerCorpusFormatter
from datasets.formatter.suc_formatter import SUCFormatter


def main(args):
    """
    - read reshaped text files
    - analyze data and output class distribution
    --------------------------------------------------------------------------------
    :return: -
    """
    if args.ner_dataset == 'swedish_ner_corpus':
        formatter = SwedishNerCorpusFormatter()
    elif args.ner_dataset == 'SUC':
        formatter = SUCFormatter()
    else:
        raise Exception(f'ner_dataset = {args.ner_dataset} unknown.')

    num_sentences = 0
    stats_aggregated = None
    for phase in ['train', 'valid']:
        _num_sentences, _stats_aggregated = formatter.read_formatted_csv(phase)
        num_sentences += _num_sentences
        if stats_aggregated is None:
            stats_aggregated = _stats_aggregated
        else:
            stats_aggregated = stats_aggregated + _stats_aggregated

        if args.verbose:
            print()
            print(f'>>> {phase} <<<<')
            print(f'num_sentences = {_num_sentences}')
            print('stats_aggregated:')
            print(_stats_aggregated)

    print()
    print(f'>>> total <<<<')
    print(f'num_sentences = {num_sentences}')
    print('stats_aggregated:')
    print(stats_aggregated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--verbose', action='store_true')
    _args = parser.parse_args()

    main(_args)
