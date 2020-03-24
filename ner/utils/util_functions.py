
import os
from ner.utils.env_variable import ENV_VARIABLE

from os.path import abspath, dirname, join
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))


def get_available_datasets():
    """
    get datasets that are available in DIR_DATASETS directory
    ---------------------------------------------------------
    :return: available datasets: [list] of [str], e.g. ['SUC', 'swedish_ner_corpus']
    """
    dir_datasets = ENV_VARIABLE['DIR_DATASETS']
    return [
        folder
        for folder in os.listdir(dir_datasets)
        if os.path.isdir(join(dir_datasets, folder))
    ]


def get_dataset_path(dataset):
    """
    get dataset path for dataset
    ----------------------------
    :param dataset:        [str] dataset name, e.g. 'SUC', 'swedish_ner_corpus'
    :return: dataset_path: [str] path to dataset directory
    """
    dir_datasets = ENV_VARIABLE['DIR_DATASETS']

    if dataset == 'SUC':
        dataset_path = join(BASE_DIR, f'{dir_datasets}/SUC/')
    elif dataset == 'swedish_ner_corpus':
        dataset_path = join(BASE_DIR, f'{dir_datasets}/swedish_ner_corpus/')
    else:
        raise Exception(f'dataset = {dataset} unknown.')

    return dataset_path


def get_rid_of_special_tokens(tag_list):
    """
    replace special tokens ('[CLS]', '[SEP]', '[PAD]') by 'O'
    ---------------------------------------------------------
    :param tag_list:           [list] of [str], e.g. ['[CLS]', 'O', 'ORG', 'ORG', '[SEP]']
    :return: cleaned_tag_list: [list] of [str], e.g. [    'O', 'O', 'ORG', 'ORG',     'O']
    """
    return [tag
            if not tag.startswith('[')
            else 'O'
            for tag in tag_list
            ]


def add_bio_to_tag_list(tag_list):
    """
    adds bio prefixes to tags
    ---------------------------
    :param tag_list:      [list] of [str], e.g. ['O',   'ORG',   'ORG']
    :return: bio_tag_list [list] of [str], e.g. ['O', 'B-ORG', 'I-ORG']
    """
    return [_add_bio_to_tag(tag_list[i], previous=tag_list[i - 1] if i > 0 else None)
            for i in range(len(tag_list))]


def _add_bio_to_tag(tag, previous):
    """
    add bio prefix to tag, depending on previous tag
    ------------------------------------------------
    :param tag:       [str], e.g. 'ORG'
    :param previous:  [str], e.g. 'ORG'
    :return: bio_tag: [str], e.g. 'I-ORG'
    """
    if tag == 'O' or tag.startswith('['):
        return tag
    elif previous is None:
        return f'B-{tag}'
    elif tag != previous:
        return f'B-{tag}'
    else:
        return f'I-{tag}'
