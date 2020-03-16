
import argparse
import torch
import mlflow

import warnings
warnings.filterwarnings('ignore')

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)

from utils.env_variable import ENV_VARIABLE
import utils.bert_ner_single as bert_ner_single
from experiment_hyperparameter_configs.experiment_hyperparameter_config import ExperimentHyperparameterConfig


def main(params, log_dirs):
    """
    :param params:   [argparse.Namespace] attr: experiment_name, run_name, experiment_run_name, device, fp16
    :param log_dirs: [argparse.Namespace] attr: mlflow, tensorboard
    :return: -
    """
    experiment_hyperparameter_config = ExperimentHyperparameterConfig(experiment_name=params.experiment_name,
                                                                      run_name=params.run_name)
    runs, _params_configs, _hparams_configs = experiment_hyperparameter_config.parse()

    with mlflow.start_run(run_name=params.experiment_name):

        for k, v in experiment_hyperparameter_config.get_params_and_hparams(run_name=None).items():
            mlflow.log_param(k, v)

        for run in runs:
            # params
            params_dict = {
                'experiment_name': params.experiment_name,
                'device': params.device,
                'fp16': params.fp16,
            }
            params_dict.update(_params_configs[run])
            params = argparse.Namespace(**params_dict)

            # hparams
            hparams = argparse.Namespace(**_hparams_configs[run])

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
    }
    _log_dirs = argparse.Namespace(**_log_dirs_dict)

    # main
    main(_params, _log_dirs)
