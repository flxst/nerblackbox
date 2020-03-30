
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
        dataset_path = join(BASE_DIR, f'{dir_datasets}/SUC')
    elif dataset == 'swedish_ner_corpus':
        dataset_path = join(BASE_DIR, f'{dir_datasets}/swedish_ner_corpus')
    elif dataset == 'conll2003':
        dataset_path = join(BASE_DIR, f'{dir_datasets}/conll2003')
    else:
        raise Exception(f'dataset = {dataset} unknown.')

    return dataset_path
