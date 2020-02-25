
import numpy as np
import pandas as pd

import argparse

import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import get_dataset_path
from datasets.formatter.swedish_ner_corpus_formatter import SwedishNerCorpusFormatter
from datasets.formatter.suc_formatter import SUCFormatter


########################################################################################################################
# ROWS
########################################################################################################################
def read_rows(_formatter, phases):
    rows = list()
    for phase in phases:
        rows.extend(_formatter.read_original_file(phase))
    return rows


def write_rows_train(_formatter, _dataset_path, _rows):
    _formatter.write_formatted_csv('train', _rows, dataset_path=_dataset_path)


def write_rows_valid_test(_formatter, _dataset_path, _rows, _valid_fraction):
    split_index = int(len(_rows) * _valid_fraction)
    rows_valid = _rows[:split_index]
    rows_test = _rows[split_index:]
    _formatter.write_formatted_csv('valid', rows_valid, dataset_path=_dataset_path)
    _formatter.write_formatted_csv('test', rows_test, dataset_path=_dataset_path)


########################################################################################################################
# CSVS
########################################################################################################################
def read_csvs(_formatter, phases):
    csv_phases = [_formatter.read_original_file(phase) for phase in phases]
    return pd.concat(csv_phases, ignore_index=True)


def write_csvs_train(_formatter, _dataset_path, _csvs):
    _formatter.write_formatted_csv('train', _csvs, dataset_path=_dataset_path)


def write_csvs_valid_test(_formatter, _dataset_path, _csvs, _valid_fraction):
    split_index = int(len(_csvs) * _valid_fraction)
    csvs_valid = _csvs.iloc[:split_index]
    csvs_test = _csvs.iloc[split_index:]
    _formatter.write_formatted_csv('valid', csvs_valid, dataset_path=_dataset_path)
    _formatter.write_formatted_csv('test', csvs_test, dataset_path=_dataset_path)


########################################################################################################################
# MAIN
########################################################################################################################
def main(args):
    """
    - read original text files
    - split valid/test
    - reshape text files to standard format
    --------------------------------------------------------------------------------
    :return: -
    """
    dataset_path = os.path.join(BASE_DIR, get_dataset_path(args.ner_dataset))

    if args.ner_dataset == 'swedish_ner_corpus':
        # formatter
        formatter = SwedishNerCorpusFormatter()

        # train -> train
        rows = read_rows(formatter, ['train'])
        write_rows_train(formatter, dataset_path, rows)

        # test  -> valid (& test)
        rows = read_rows(formatter, ['test'])
        write_rows_valid_test(formatter, dataset_path, rows, args.valid_fraction)
    elif args.ner_dataset == 'SUC':
        # formatter
        formatter = SUCFormatter()

        # valid, test -> train
        csvs = read_csvs(formatter, ['valid', 'test'])
        write_csvs_train(formatter, dataset_path, csvs)

        # train       -> valid (& test)
        csvs = read_csvs(formatter, ['train'])
        write_csvs_valid_test(formatter, dataset_path, csvs, args.valid_fraction)
    else:
        raise Exception(f'ner_dataset = {args.ner_dataset} unknown.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--valid_fraction', type=float, default=1.0)
    _args = parser.parse_args()

    main(_args)
