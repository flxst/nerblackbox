
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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
    label_list = processor.get_label_list()

    # train data
    input_examples_train_all = processor.get_input_examples('train')
    input_examples['train'] = prune_examples(input_examples_train_all, ratio=prune_ratio[0])

    # validation data
    input_examples_valid_all = processor.get_input_examples('valid')
    input_examples['valid'] = prune_examples(input_examples_valid_all, ratio=prune_ratio[1])

    # input_examples_to_tensors
    input_examples_to_tensors = InputExampleToTensors(tokenizer,
                                                      max_seq_length=max_seq_length,
                                                      label_tuple=tuple(label_list))

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

    return dataloader, label_list


def get_available_models():
    """ TODO: TEMPORARY METHOD: GET RID OF THIS"""
    d = ENV_VARIABLE['DIR_PRETRAINED_MODELS']
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


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


def get_rid_of_special_tokens(label_list):
    """
    replace special tokens ('[CLS]', '[SEP]', '[PAD]') by 'O'
    ---------------------------------------------------------
    :param label_list:           [list] of [str], e.g. ['[CLS]', 'O', 'ORG', 'ORG', '[SEP]']
    :return: cleaned_label_list: [list] of [str], e.g. [    'O', 'O', 'ORG', 'ORG',     'O']
    """
    return [label
            if not label.startswith('[')
            else 'O'
            for label in label_list
            ]


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


def display_available_metrics():
    # basic & hyperparameters
    columns = ['dataset', 'pretrained_model_name', 'num_epochs', 'prune_ratio', 'lr_schedule']

    files = [file for file in os.listdir(ENV_VARIABLE['DIR_CHECKPOINTS']) if file.endswith('.pkl')]
    files_metrics = [file for file in files if file.startswith('metrics')]
    files_metrics_tup = [file.replace('.pkl', '').split('__')[1:] for file in files_metrics]

    df = pd.DataFrame(files_metrics_tup, columns=columns)
    df['num_epochs'] = df['num_epochs'].astype(int)
    df['prune_ratio'] = df['prune_ratio'].astype(float)

    # metrics
    columns_metrics = ['f1_macro_all', 'f1_micro_all', 'f1_macro_fil', 'f1_micro_fil']
    data_metrics = []
    for file in files_metrics:
        data_metrics_row = []
        with open(os.path.join(ENV_VARIABLE['DIR_CHECKPOINTS'], file), 'rb') as f:
            metrics = pickle.load(f)
            data_metrics_row.append(metrics['epoch']['valid']['f1']['macro']['all'][-1])
            data_metrics_row.append(metrics['epoch']['valid']['f1']['micro']['all'][-1])
            data_metrics_row.append(metrics['epoch']['valid']['f1']['macro']['fil'][-1])
            data_metrics_row.append(metrics['epoch']['valid']['f1']['micro']['fil'][-1])

        data_metrics.append(data_metrics_row)
        
    # print(data_metrics)

    df_metrics = pd.DataFrame(data_metrics, columns=columns_metrics)
    # print(df_metrics)

    return pd.concat([df, df_metrics], axis=1).sort_values(by=columns).reset_index(drop=True)


########################################################################################################################
# PLOT
########################################################################################################################
def load_and_plot_metrics(pick):
    # LOAD #
    columns = ['dataset', 'pretrained_model_name', 'num_epochs', 'prune_ratio', 'lr_schedule']
    _pick = {k: pick[k] for k in columns}
    metrics = load_metrics(**_pick)

    # PLOT #
    # display(metrics)

    plot_learning_rate(metrics)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_metric(metrics, pick['num_epochs'], 'loss', ax=ax[0])
    plot_metric(metrics, pick['num_epochs'], 'acc', ax=ax[1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_metric(metrics, pick['num_epochs'], 'f1', ('macro', 'all'), ax=ax[0])
    plot_metric(metrics, pick['num_epochs'], 'f1', ('macro', 'fil'), ax=ax[1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_metric(metrics, pick['num_epochs'], 'f1', ('micro', 'all'), ax=ax[0])
    plot_metric(metrics, pick['num_epochs'], 'f1', ('micro', 'fil'), ax=ax[1])


def load_metrics(dataset, pretrained_model_name, num_epochs, prune_ratio, lr_schedule):

    dir_checkpoints = ENV_VARIABLE['DIR_CHECKPOINTS']

    model_name = pretrained_model_name.split('/')[-1]
    if lr_schedule is None:
        pkl_path = f'./{dir_checkpoints}/metrics__{dataset}__{model_name}__{num_epochs}__{prune_ratio}.pkl'
    else:
        pkl_path = f'./{dir_checkpoints}/metrics__{dataset}__{model_name}__{num_epochs}__{prune_ratio}__{lr_schedule}.pkl'
    with open(pkl_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


def display(_metrics):
    print('--- train ---')
    print('> batch')
    print(_metrics['batch']['train'])
    print('--- valid ---')
    print('> batch')
    print(_metrics['batch']['valid'])
    print('> epoch')
    print(_metrics['epoch']['valid'])


def plot_learning_rate(metrics):
    lr = metrics['batch']['train']['lr']
    fig, ax = plt.subplots()
    ax.plot(lr, linestyle='', marker='.')
    ax.set_xlabel('batch')
    ax.set_ylabel('learning rate')


def plot_metric(metrics, num_epochs, metric, f1_spec=None, ax=None):
    # PREP #
    if f1_spec is None:
        batch_train = metrics['batch']['train'][metric]
        epoch_valid = metrics['epoch']['valid'][metric]
    else:
        batch_train = metrics['batch']['train'][metric][f1_spec[0]][f1_spec[1]]
        epoch_valid = metrics['epoch']['valid'][metric][f1_spec[0]][f1_spec[1]]

    clr = {'loss': 'r',
           'acc': 'green',
           'f1_macro': 'orange',
           'f1_micro': 'blue',
           }
    if f1_spec is None:
        metric_spec = metric
    else:
        f1_spec_1st = f1_spec[0]
        metric_spec = f'{metric}_{f1_spec_1st}'

    # PLOT #
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(batch_train,
            linestyle='-', marker='.', color=clr[metric_spec], alpha=0.3, label='train')

    x = [len(batch_train) * float(i) / num_epochs for i in range(1, num_epochs + 1)]
    ax.plot(x, epoch_valid,
            linestyle='', marker='o', color=clr[metric_spec], label='valid')

    ax.set_xlabel('batch')
    ax.set_ylabel(metric)
    if metric == 'loss':
        ax.set_ylim([0, None])
    else:
        ax.set_ylim([0, 1])
    if metric in ['loss', 'acc']:
        ax.set_title(metric)
    elif metric == 'f1':
        f1_spec_1st = f1_spec[0]
        f1_spec_2nd = f1_spec[1]
        ax.set_title(f'f1 score: {f1_spec_1st}, {f1_spec_2nd}')
    ax.legend()
