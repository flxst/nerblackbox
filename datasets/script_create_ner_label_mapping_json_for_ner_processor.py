
import argparse
import json

import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from datasets.formatter.swedish_ner_corpus_formatter import SwedishNerCorpusFormatter
from datasets.formatter.suc_formatter import SUCFormatter


def main(args):
    """
    writes ner_label_mapping.json file
    ----------------------------------
    :param args: [argparse parsed arguments]
        ner_dataset: [str], e.g. 'swedish_ner_corpus'
        with_tags:   [bool]. If true, have tags like 'B-PER', 'I-PER'. If false, have tags like 'PER'.
        modify:      [bool], if True: modify labels as specified in method modify_ner_label_mapping()
    :return: -
    """
    # formatter
    if args.ner_dataset == 'swedish_ner_corpus':
        formatter = SwedishNerCorpusFormatter()
    elif args.ner_dataset == 'SUC':
        formatter = SUCFormatter()
    else:
        raise Exception(f'ner_dataset = {args.ner_dataset} unknown.')

    # ner label mapping
    ner_label_mapping = formatter.create_ner_label_mapping(with_tags=args.with_tags, modify=args.modify)

    json_path = f'datasets/ner/{args.ner_dataset}/ner_label_mapping.json'
    with open(json_path, 'w') as f:
        json.dump(ner_label_mapping, f)

    print(f'> dumped the following dict to {json_path}:')
    print(ner_label_mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--with_tags', action='store_true')
    parser.add_argument('--modify', action='store_true')
    _args = parser.parse_args()

    main(_args)
