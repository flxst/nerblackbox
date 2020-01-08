
import os

ENV_VARIABLE = {
    'DIR_PRETRAINED_MODELS': './pretrained_models',
    'DIR_DATASETS': './datasets',
    'DIR_CHECKPOINTS': './checkpoints',
}


def get_available_models():
    d = ENV_VARIABLE['DIR_PRETRAINED_MODELS']
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


def get_available_datasets(dataset_type):
    d = os.path.join(ENV_VARIABLE['DIR_DATASETS'], dataset_type)
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


def prune_examples(_examples, ratio=None):
    if ratio is None:
        return _examples
    else:
        num_examples_new = int(ratio*float(len(_examples)))
        print(f'use {num_examples_new} of {len(_examples)} examples')
        return _examples[:num_examples_new]
