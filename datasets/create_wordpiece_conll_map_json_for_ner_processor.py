
import argparse
import json


def main(ner_dataset):
    if ner_dataset == 'swedish_ner_corpus':
        wordpiece_conll_map = {
            'O': 'O',
            'PER': 'PER',
            'ORG': 'ORG',
            'LOC': 'LOC',
            'MISC': 'MISC',
            'ORG*': 'ORG',
        }
    else:
        raise Exception(f'ner_dataset = {ner_dataset} unknown.')

    json_path = f'datasets/ner/{ner_dataset}/wordpiece_conll_map.json'
    with open(json_path, 'w') as f:
        json.dump(wordpiece_conll_map, f)

    print(f'> dumped the following dict to {json_path}:')
    print(wordpiece_conll_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, help='model name')
    args = parser.parse_args()

    main(args.model)
