
import os
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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def main(args):
    ####################################################################################################################
    # mlflow
    ####################################################################################################################
    # set env variable experiment name
    os.environ['MLFLOW_EXPERIMENT_NAME'] = args.experiment_name if args.experiment_name is not None else 'DEFAULT'
    os.environ['MLFLOW_RUN_NAME'] = args.run_name if args.run_name is not None else 'DEFAULT'

    try:
        print('mlflow experiment_name:', os.environ['MLFLOW_EXPERIMENT_NAME'])
        print('mlflow run_name:       ', os.environ['MLFLOW_RUN_NAME'])
    except:
        pass

    ####################################################################################################################
    # device
    ####################################################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    fp16 = True if args.fp16 and device == 'cuda' else False
    print(f'> Available GPUs: {torch.cuda.device_count()}')
    print(f'> Using device:   {device}')
    print(f'> Using fp16:     {fp16}')
    print('---------------------------')

    ####################################################################################################################
    # hyperparameters
    ####################################################################################################################
    hyperparams = {
        'device': device.type,
        'fp16': fp16,
        'batch_size': args.batch_size,
        'max_seq_length': args.max_seq_length,
        'num_epochs': args.num_epochs,
        'prune_ratio': (args.prune_ratio_train, args.prune_ratio_valid),
        'learning_rate': {
            'lr_max': args.lr_max,
            'lr_schedule': args.lr_schedule,
            'lr_warmup_fraction': args.lr_warmup_fraction,
            'lr_num_cycles': args.lr_num_cycles,
        },
    }

    ####################################################################################################################
    # START
    ####################################################################################################################
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=False)  # needs to be False !!

    # data
    dataset_path = os.path.join(BASE_DIR, get_dataset_path(args.dataset_name))
    dataloader, tag_list = preprocess_data(dataset_path,
                                           tokenizer,
                                           hyperparams['batch_size'],
                                           max_seq_length=hyperparams['max_seq_length'],
                                           prune_ratio=hyperparams['prune_ratio']
                                           )

    # model
    model = BertForTokenClassification.from_pretrained(args.pretrained_model_name,
                                                       num_labels=len(tag_list))

    # trainer
    trainer = NERTrainer(model,
                         device,
                         train_dataloader=dataloader['train'],
                         valid_dataloader=dataloader['valid'],
                         tag_list=tag_list,
                         hyperparams=hyperparams,
                         fp16=fp16,
                         verbose=False,
                         )
    trainer.fit(num_epochs=hyperparams['num_epochs'],
                **hyperparams['learning_rate'],
                )

    ####################################################################################################################
    # MANUAL MODELS & METRICS SAVING
    ####################################################################################################################
    if args.checkpoints:
        trainer.save_model_checkpoint(args.dataset_name,
                                      args.pretrained_model_name,
                                      hyperparams['num_epochs'],
                                      hyperparams['prune_ratio'][0],
                                      hyperparams['learning_rate']['lr_schedule'],
                                      )

        trainer.save_metrics(args.dataset_name,
                             args.pretrained_model_name,
                             hyperparams['num_epochs'],
                             hyperparams['prune_ratio'][0],
                             hyperparams['learning_rate']['lr_schedule'],
                             )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mlflow
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    # model & dataset
    parser.add_argument('--pretrained_model_name', type=str, default='af-ai-center/bert-base-swedish-uncased')
    parser.add_argument('--dataset_name', type=str, default='swedish_ner_corpus')
    # device
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--fp16', type=bool, default=False)
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--prune_ratio_train', type=float, default=0.01)
    parser.add_argument('--prune_ratio_valid', type=float, default=0.01)
    parser.add_argument('--lr_max', type=float, default=2e-5)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--lr_warmup_fraction', type=float, default=0.1)
    parser.add_argument('--lr_num_cycles', type=float, default=None)
    # additional
    parser.add_argument('--checkpoints', action='store_true')

    _args = parser.parse_args()
    main(_args)
