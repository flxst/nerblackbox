
import argparse
import torch
import mlflow

import logging
logging.basicConfig(level=logging.WARNING)  # basic setting that is mainly applied to mlflow's default logging

import warnings
warnings.filterwarnings('ignore')

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
sys.path.append(BASE_DIR)

from ner_black_box.utils.env_variable import ENV_VARIABLE
import ner_black_box.ner_training.bert_ner_single as bert_ner_single
from ner_black_box.experiment_config import ExperimentConfig


def main(params, log_dirs):
    """
    :param params:   [argparse.Namespace] attr: experiment_name, run_name, device, fp16, experiment_run_name
    :param log_dirs: [argparse.Namespace] attr: mlflow, tensorboard
    :return: -
    """
    experiment_config = ExperimentConfig(experiment_name=params.experiment_name,
                                         run_name=params.run_name,
                                         device=params.device,
                                         fp16=params.fp16)
    runs, run_params, run_hparams = experiment_config.parse()

    with mlflow.start_run(run_name=params.experiment_name):
        for k, v in experiment_config.get_params_and_hparams(run_name=None).items():
            mlflow.log_param(k, v)

        for run in runs:
            # params & hparams: dict -> namespace
            params = argparse.Namespace(**run_params[run])
            hparams = argparse.Namespace(**run_hparams[run])

            # bert_ner: single run
            bert_ner_single.main(params, hparams, log_dirs, experiment=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params
    args_general = parser.add_argument_group('args_general')
    args_general.add_argument('--experiment_name', type=str, required=True)  # .. logging w/ mlflow & tensorboard
    args_general.add_argument('--run_name', type=str, required=True)         # .. logging w/ mlflow & tensorboard
    args_general.add_argument('--device', type=str, required=True)           # .. device
    args_general.add_argument('--fp16', type=bool, required=True)            # .. device

    # parsing
    _args = parser.parse_args()
    _params = None
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(_args, a.dest, None) for a in group._group_actions}
        if group.title == 'args_general':
            group_dict['device'] = torch.device('cuda' if torch.cuda.is_available() and group_dict['device'] == 'gpu' else 'cpu')
            group_dict['fp16'] = True if group_dict['fp16'] and group_dict['device'] == 'cuda' else False
            if len(group_dict['run_name']) == 0:
                group_dict['run_name'] = None
            _params = argparse.Namespace(**group_dict)

    # log_dirs
    _log_dirs_dict = {
        'mlflow': ENV_VARIABLE['DIR_MLFLOW'],
        'tensorboard': ENV_VARIABLE['DIR_TENSORBOARD'],
        'checkpoints': ENV_VARIABLE['DIR_CHECKPOINTS'],
        'log_file': ENV_VARIABLE['LOG_FILE'],
        'mlflow_file': ENV_VARIABLE['MLFLOW_FILE'],
    }
    _log_dirs = argparse.Namespace(**_log_dirs_dict)

    # main
    main(_params, _log_dirs)
