
import os
from os.path import abspath, dirname
import logging
import argparse

import torch
from transformers import BertTokenizer, BertForTokenClassification

from utils.utils import preprocess_data
from utils.utils import get_dataset_path
from utils.ner_trainer import NERTrainer

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

BASE_DIR = abspath(dirname(__file__))


def main(params, hparams):
    ####################################################################################################################
    # mlflow
    ####################################################################################################################
    # set env variable experiment name
    os.environ['MLFLOW_EXPERIMENT_NAME'] = params.experiment_name
    os.environ['MLFLOW_RUN_NAME'] = params.run_name
    experiment_run_name = f'{params.experiment_name}/{params.run_name}'
    print('mlflow experiment_name:     ', params.experiment_name)
    print('mlflow run_name:            ', params.run_name)
    print('mlflow experiment_run_name: ', experiment_run_name)

    ####################################################################################################################
    # device
    ####################################################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() and params.device == 'gpu' else 'cpu')
    fp16 = True if params.fp16 and device == 'cuda' else False
    print(f'> Available GPUs: {torch.cuda.device_count()}')
    print(f'> Using device:   {device}')
    print(f'> Using fp16:     {fp16}')
    print('---------------------------')

    ####################################################################################################################
    # hyperparameters
    ####################################################################################################################
    hyperparameters = vars(hparams)

    ####################################################################################################################
    # START
    ####################################################################################################################
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(params.pretrained_model_name, do_lower_case=False)  # needs to be False !!

    # data
    dataset_path = os.path.join(BASE_DIR, get_dataset_path(params.dataset_name))
    dataloader, tag_list = preprocess_data(dataset_path,
                                           tokenizer,
                                           hyperparameters['batch_size'],
                                           max_seq_length=hyperparameters['max_seq_length'],
                                           prune_ratio=(hyperparameters['prune_ratio_train'],
                                                        hyperparameters['prune_ratio_valid'])
                                           )

    # model
    model = BertForTokenClassification.from_pretrained(params.pretrained_model_name,
                                                       num_labels=len(tag_list))

    # trainer
    trainer = NERTrainer(model,
                         device,
                         train_dataloader=dataloader['train'],
                         valid_dataloader=dataloader['valid'],
                         tag_list=tag_list,
                         hyperparams=hyperparameters,
                         fp16=fp16,
                         verbose=False,
                         )
    trainer.fit(max_epochs=hyperparameters['max_epochs'],
                lr_max=hyperparameters['lr_max'],
                lr_schedule=hyperparameters['lr_schedule'],
                lr_warmup_fraction=hyperparameters['lr_warmup_fraction'],
                lr_num_cycles=hyperparameters['lr_num_cycles'],
                )

    ####################################################################################################################
    # MANUAL MODELS & METRICS SAVING
    ####################################################################################################################
    if params.checkpoints:
        trainer.save_model_checkpoint(params.dataset_name,
                                      params.pretrained_model_name,
                                      hyperparameters['max_epochs'],
                                      hyperparameters['prune_ratio'][0],
                                      hyperparameters['lr_schedule'],
                                      )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params
    args_general = parser.add_argument_group('args_general')
    # .. mlflow
    args_general.add_argument('--experiment_name', type=str, default='Default')
    args_general.add_argument('--run_name', type=str, default='Default')
    # .. model & dataset
    args_general.add_argument('--pretrained_model_name', type=str, default='af-ai-center/bert-base-swedish-uncased')
    args_general.add_argument('--dataset_name', type=str, default='swedish_ner_corpus')
    # .. device
    args_general.add_argument('--device', type=str, default='gpu')
    args_general.add_argument('--fp16', type=bool, default=False)
    # .. additional
    args_general.add_argument('--checkpoints', action='store_true')

    # hparams
    args_hparams = parser.add_argument_group('args_hparams')
    args_hparams.add_argument('--batch_size', type=int, default=16)
    args_hparams.add_argument('--max_seq_length', type=int, default=64)
    args_hparams.add_argument('--max_epochs', type=int, default=2)
    args_hparams.add_argument('--prune_ratio_train', type=float, default=0.01)
    args_hparams.add_argument('--prune_ratio_valid', type=float, default=0.01)
    args_hparams.add_argument('--lr_max', type=float, default=2e-5)
    args_hparams.add_argument('--lr_schedule', type=str, default='constant')
    args_hparams.add_argument('--lr_warmup_fraction', type=float, default=0.1)
    args_hparams.add_argument('--lr_num_cycles', type=float, default=4)

    # parsing
    _args = parser.parse_args()

    _params, _hparams = None, None
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(_args, a.dest, None) for a in group._group_actions}
        if group.title == 'args_general':
            _params = argparse.Namespace(**group_dict)
        elif group.title == 'args_hparams':
            _hparams = argparse.Namespace(**group_dict)

    main(_params, _hparams)
