
import argparse

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
sys.path.append(BASE_DIR)

from ner_black_box.datasets.formatter.custom_formatter import CustomFormatter


def main(args):
    """
    - V: analyze data
      . read reshaped text files
      . analyze data and output class distribution
    --------------------------------------------------------------------------------
    :return: -
    """
    formatter = CustomFormatter.for_dataset(args.ner_dataset)
    formatter.analyze_data()                   # IV: analyze data
    formatter.plot_data()                      # IV: analyze data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', required=True, type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--verbose', action='store_true')
    _args = parser.parse_args()

    main(_args)
