
import argparse
import subprocess

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)


def main(args):
    """
    - get data for ner_dataset
    --------------------------------------------------------------------------------
    :return: -
    """
    if args.ner_dataset == 'swedish_ner_corpus':
        bash_cmd = 'git clone https://github.com/klintan/swedish-ner-corpus.git datasets/ner/swedish_ner_corpus'

        if args.verbose:
            print(bash_cmd)

        try:
            subprocess.run(bash_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
    elif args.ner_dataset == 'SUC':
        print('(SUC: nothing to do)')
    else:
        raise Exception(f'ner_dataset = {args.ner_dataset} unknown.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_dataset', required=True, type=str, help='e.g. swedish_ner_corpus')
    parser.add_argument('--verbose', action='store_true')
    _args = parser.parse_args()

    main(_args)
