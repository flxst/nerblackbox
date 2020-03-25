
import argparse

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.custom_formatter import CustomFormatter


def main(args):
    """
    - I: get data for ner_dataset
    - II: write ner_tag_mapping.json file
    - III: format data
      . reshape original files to standard format
    - IV: resplit data
      . split train/valid/test
    - V: analyze data
      . read reshaped text files
      . analyze data and output class distribution
    --------------------------------------------------------------------------------
    :return: -
    """
    # formatter
    formatter = CustomFormatter.for_dataset(args.ner_dataset)

    formatter.create_directory()
    formatter.get_data(verbose=args.verbose)                                        # I: get_data
    formatter.create_ner_tag_mapping(with_tags=args.with_tags, modify=args.modify)  # II: create ner tag mapping
    formatter.format_data()                                                         # III: format data
    formatter.resplit_data(valid_fraction=args.valid_fraction)                      # IV: resplit data
    formatter.analyze_data()                                                        # V: analyze data
    formatter.plot_data()                                                           # V: analyze data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', required=True, type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--with_tags', type=bool, default=False)
    parser.add_argument('--modify', type=bool, default=True)
    parser.add_argument('--valid_fraction', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')
    _args = parser.parse_args()

    main(_args)
