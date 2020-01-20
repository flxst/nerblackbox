
import os

ENV_VARIABLE = {
    'DIR_PRETRAINED_MODELS': './pretrained_models',  # TODO: get rid of this line
    'DIR_DATASETS': './datasets',
    'DIR_CHECKPOINTS': './checkpoints',
}


def get_available_models():
    """ TODO: TEMPORARY METHOD: GET RID OF THIS"""
    d = ENV_VARIABLE['DIR_PRETRAINED_MODELS']
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


def get_available_datasets(downstream_task):
    """
    get datasets that are available in DIR_DATASETS directory
    ---------------------------------------------------------
    :param downstream_task: [str] e.g. 'ner'
    :return: available datasets: [list] of [str], e.g. ['SUC', 'swedish_ner_corpus']
    """
    d = os.path.join(ENV_VARIABLE['DIR_DATASETS'], downstream_task)
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


def prune_examples(list_of_examples, ratio=None):
    """
    prunes list_of_examples by taking only the first examples
    ---------------------------------------------------------
    :param list_of_examples: [list], e.g. of [InputExample]
    :param (Optional) ratio: [float], e.g. 0.5
    :return: [list], e.g. of [InputExample]
    """
    if ratio is None:
        return list_of_examples
    else:
        num_examples_old = len(list_of_examples)
        num_examples_new = int(ratio*float(num_examples_old))
        print(f'use {num_examples_new} of {num_examples_old} examples')
        return list_of_examples[:num_examples_new]


def add_bio_to_label_list(label_list):
    """
    adds bio prefixes to labels
    ---------------------------
    :param label_list:      [list] of [str], e.g. ['O',   'ORG',   'ORG']
    :return: bio_label_list [list] of [str], e.g. ['O', 'B-ORG', 'I-ORG']
    """
    return [_add_bio_to_label(label_list[i], previous=label_list[i - 1] if i > 0 else None)
            for i in range(len(label_list))]


def _add_bio_to_label(label, previous):
    """
    add bio prefix to label, depending on previous label
    ----------------------------------------------------
    :param label:       [str], e.g. 'ORG'
    :param previous:    [str], e.g. 'ORG'
    :return: bio_label: [str], e.g. 'I-ORG'
    """
    if label == 'O' or label.startswith('['):
        return label
    elif previous is None:
        return f'B-{label}'
    elif label != previous:
        return f'B-{label}'
    else:
        return f'I-{label}'
