
import os

from utils.bert_dataset import BertDataset
from utils.ner_processor import NerProcessor
from utils.input_example_to_tensors import InputExampleToTensors
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.env_variable import ENV_VARIABLE


def preprocess_data(dataset_path, tokenizer, batch_size, max_seq_length=64, prune_ratio=(1.0, 1.0)):
    input_examples = dict()
    data = dict()
    dataloader = dict()

    # processor
    processor = NerProcessor(dataset_path, tokenizer, do_lower_case=True)  # needs to be True (applies .lower()) !!
    tag_list = processor.get_tag_list()

    # train data
    input_examples_train_all = processor.get_input_examples('train')
    input_examples['train'] = prune_examples(input_examples_train_all, ratio=prune_ratio[0])

    # validation data
    input_examples_valid_all = processor.get_input_examples('valid')
    input_examples['valid'] = prune_examples(input_examples_valid_all, ratio=prune_ratio[1])

    # input_examples_to_tensors
    input_examples_to_tensors = InputExampleToTensors(tokenizer,
                                                      max_seq_length=max_seq_length,
                                                      tag_tuple=tuple(tag_list))

    # dataloader
    data['train'] = BertDataset(input_examples['train'],
                                transform=input_examples_to_tensors)
    dataloader['train'] = DataLoader(data['train'],
                                     sampler=RandomSampler(data['train']),
                                     batch_size=batch_size)

    data['valid'] = BertDataset(input_examples['valid'],
                                transform=input_examples_to_tensors)
    dataloader['valid'] = DataLoader(data['valid'],
                                     sampler=SequentialSampler(data['valid']),
                                     batch_size=batch_size)

    return dataloader, tag_list


def get_available_datasets():
    """
    get datasets that are available in DIR_DATASETS directory
    ---------------------------------------------------------
    :return: available datasets: [list] of [str], e.g. ['SUC', 'swedish_ner_corpus']
    """
    d = os.path.join(ENV_VARIABLE['DIR_DATASETS'])
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
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
        dataset_path = f'{dir_datasets}/SUC/'
    elif dataset == 'swedish_ner_corpus':
        dataset_path = f'{dir_datasets}/swedish_ner_corpus/'
    else:
        raise Exception(f'dataset = {dataset} unknown.')

    return dataset_path


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
