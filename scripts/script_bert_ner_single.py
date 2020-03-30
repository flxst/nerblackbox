
import argparse
import torch

import warnings
warnings.filterwarnings('ignore')

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)

from ner.utils.env_variable import ENV_VARIABLE
import ner.bert_ner_single as bert_ner_single


def main(params, hparams, log_dirs):
    """
    :param params:   [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param hparams:  [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param log_dirs: [argparse.Namespace] attr: mlflow, tensorboard
    :return: -
    """
    bert_ner_single.main(params, hparams, log_dirs, experiment=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params
    args_general = parser.add_argument_group('args_general')
    args_general.add_argument('--experiment_name', type=str, default='Default')  # .. logging w/ mlflow & tensorboard
    args_general.add_argument('--run_name', type=str, default='Default')         # .. logging w/ mlflow & tensorboard
    args_general.add_argument('--pretrained_model_name', type=str, default='af-ai-center/bert-base-swedish-uncased')
    args_general.add_argument('--dataset_name', type=str, default='swedish_ner_corpus')          # .. model & dataset
    args_general.add_argument('--device', type=str, default='gpu')                               # .. device
    args_general.add_argument('--fp16', type=bool, default=False)                                # .. device
    args_general.add_argument('--checkpoints', action='store_true')                              # .. checkpoints
    args_general.add_argument('--monitor', type=str, default='val_loss')                         # .. early stopping
    args_general.add_argument('--min_delta', type=float, default=0.00)                           # .. early stopping
    args_general.add_argument('--patience', type=int, default=2)                                 # .. early stopping
    args_general.add_argument('--mode', type=str, default='min')                                 # .. early stopping

    # hparams
    args_hparams = parser.add_argument_group('args_hparams')
    args_hparams.add_argument('--batch_size', type=int, default=16)
    args_hparams.add_argument('--max_seq_length', type=int, default=64)
    args_hparams.add_argument('--prune_ratio_train', type=float, default=0.01)
    args_hparams.add_argument('--prune_ratio_val', type=float, default=0.01)
    args_hparams.add_argument('--prune_ratio_test', type=float, default=0.01)
    args_hparams.add_argument('--max_epochs', type=int, default=1)
    args_hparams.add_argument('--lr_max', type=float, default=2e-5)
    args_hparams.add_argument('--lr_warmup_epochs', type=int, default=1)
    args_hparams.add_argument('--lr_schedule', type=str, default='constant', help='constant/linear/cosine/cosine_w[..]')
    args_hparams.add_argument('--lr_num_cycles', type=float, default=4)

    # parsing
    _args = parser.parse_args()
    _params, _hparams = None, None
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(_args, a.dest, None) for a in group._group_actions}
        if group.title == 'args_general':
            group_dict['experiment_run_name'] = group_dict['experiment_name'] + '/' + group_dict['run_name']
            group_dict['device'] = torch.device('cuda' if torch.cuda.is_available() and group_dict['device'] == 'gpu' else 'cpu')
            group_dict['fp16'] = True if group_dict['fp16'] and group_dict['device'] == 'cuda' else False
            _params = argparse.Namespace(**group_dict)
        elif group.title == 'args_hparams':
            _hparams = argparse.Namespace(**group_dict)

    # log_dirs
    _log_dirs_dict = {
        'mlflow': ENV_VARIABLE['DIR_MLFLOW'],
        'tensorboard': ENV_VARIABLE['DIR_TENSORBOARD'],
        'checkpoints': ENV_VARIABLE['DIR_CHECKPOINTS'],
    }
    _log_dirs = argparse.Namespace(**_log_dirs_dict)

    # main
    main(_params, _hparams, _log_dirs)
