
import os
import argparse
import mlflow

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(__file__))
sys.path.append(BASE_DIR)

from ner.utils.env_variable import ENV_VARIABLE


def main(experiment_name,
         run_name=None,
         device='gpu',
         fp16=False):
    """
    :param experiment_name: [str], e.g. 'exp1'
    :param run_name:        [str], e.g. 'run1', OPTIONAL
    :param device:          [str], 'gpu' or 'cpu'
    :param fp16:            [bool]
    :return: -
    """

    os.environ["MLFLOW_TRACKING_URI"] = ENV_VARIABLE['DIR_MLFLOW']

    parameters = {
        'experiment_name': experiment_name,
        'run_name': run_name if run_name else '',
        'device': device,
        'fp16': fp16,
    }

    mlflow.projects.run(
        uri=BASE_DIR,
        experiment_name=experiment_name,
        parameters=parameters,
        use_conda=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params
    parser.add_argument('--experiment_name', required=True, type=str, help='e.g. exp1')
    parser.add_argument('--run_name', default=None, type=str, help='e.g. run1')
    parser.add_argument('--device', default='gpu', type=str, help='gpu or cpu')
    parser.add_argument('--fp16', default=False, type=bool, help='')
    _args = parser.parse_args()

    main(
        experiment_name=_args.experiment_name,
        run_name=_args.run_name,
        device=_args.device,
        fp16=_args.fp16,
    )
