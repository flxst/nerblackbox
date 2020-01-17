
import argparse
import json


def main(ner_dataset, with_tags=False):
    """
    writes ner_label_mapping.json file
    ----------------------------------
    :param ner_dataset: [str], e.g. 'swedish_ner_corpus'
    :param with_tags:   [bool]. If true, have tags like 'B-PER', 'I-PER'. If false, have tags like 'PER'.
    :return: -
    """
    if ner_dataset == 'swedish_ner_corpus':
        label_list = ['PER', 'ORG', 'LOC', 'MISC']

        if with_tags:
            label_lists_extended = [[f'B-{label}', f'I-{label}'] for label in label_list]
            label_lists_full = ['O'] + [l_i for l in label_lists_extended for l_i in l]
        else:
            label_lists_full = ['O'] + label_list

        # map each label to itself
        ner_label_mapping = {k: k for k in label_lists_full}

        # take care of extra case: ORG*
        if with_tags:
            ner_label_mapping['B-ORG*'] = 'B-ORG'
            ner_label_mapping['I-ORG*'] = 'I-ORG'
        else:
            ner_label_mapping['ORG*'] = 'ORG'
    else:
        raise Exception(f'ner_dataset = {ner_dataset} unknown.')

    json_path = f'datasets/ner/{ner_dataset}/ner_label_mapping.json'
    with open(json_path, 'w') as f:
        json.dump(ner_label_mapping, f)

    print(f'> dumped the following dict to {json_path}:')
    print(ner_label_mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, help='model name')
    parser.add_argument('--with_tags', action='store_true')
    args = parser.parse_args()

    main(args.model, args.with_tags)
