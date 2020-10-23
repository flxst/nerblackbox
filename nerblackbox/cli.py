r"""Command Line Interface of the nerblackbox package."""

import os
import subprocess
from os.path import join
import click
from nerblackbox.modules.main import NerBlackBoxMain


@click.command()
@click.argument('flag')
@click.argument('flag_args', nargs=-1, type=click.UNPROCESSED)
@click.option("--data_dir",             default='data', type=str,   help="[str] relative path of data directory")
@click.option("--modify",               default=None,   type=bool,  help="[bool] if flag=set_up_dataset")
@click.option("--val_fraction",         default=None,   type=float, help="[float] if flag=set_up_dataset")
@click.option("--verbose/--no-verbose", default=False,              help="[bool] if flag=set_up_dataset")
@click.option("--run_name",             default=None,   type=str,   help="[str] if flag=run_experiment")
@click.option("--device",               default=None,   type=str,   help="[str] if flag=run_experiment")
@click.option("--fp16/--no-fp16",       default=False,              help="[bool] if flag=run_experiment")
@click.option("--results/--no-results", default=False,              help="[bool] if flag=clear_data")
def main(**kwargs_optional):
    r"""

    :param FLAG:

    - analyze_data:
        analyze a dataset.

        ``FLAG_ARGS = <dataset_name>``

    - clear_data:
        clear data (checkpoints and optionally results)

    - get_experiment_results:
        get results for a single experiment.

        ``FLAG_ARGS = <experiment_name>``

    - get_experiments:
        show list of experiments that have been run.

    - get_experiments_results:
        get results from multiple experiments.

    - init:
        initialize the data_dir directory, download & prepare built-in datasets, prepare experiment configuration.
        needs to be called exactly once before any other CLI/API commands of the package are executed.

    - mlflow:
        show detailed experiment results in mlflow (port = 5000)

    - predict:
        predict labels for text_input using the best model of a single experiment.

        ``FLAG_ARGS = <experiment_name> <text_input>``

    - run_experiment:
        run a single experiment.

        ``FLAG_ARGS = <experiment_name>``

    - set_up_dataset:
        set up a dataset using the associated Formatter class.

        ``FLAG_ARGS = <dataset_name>``

    - show_experiment_config:
        show a single experiment configuration in detail.

        ``FLAG_ARGS = <experiment_name>``

    - show_experiment_configs:
        show overview on all available experiment configurations.

    - tensorboard:
        show detailed experiment results in tensorboard. (port = 6006)
    """

    # kwargs
    kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}
    data_dir = kwargs.pop('data_dir')

    possible_flags = [
        'analyze_data',
        'clear_data',
        'get_experiment_results',
        'get_experiments',
        'get_experiments_results',
        'init',
        'mlflow',
        'predict',
        'run_experiment',
        'set_up_dataset',
        'show_experiment_config',
        'show_experiment_configs',
        'tensorboard',
    ]

    # environ
    base_dir = os.getcwd()
    os.environ['BASE_DIR'] = base_dir
    os.environ['DATA_DIR'] = join(base_dir, data_dir)
    # print('BASE_DIR = ', os.environ.get('BASE_DIR'))
    # print('DATA_DIR = ', os.environ.get('DATA_DIR'))

    if 'flag' not in kwargs.keys():
        print('need to specify flag (see --help)')
        exit(0)
    elif kwargs['flag'] not in possible_flags:
        print(f'flag = {kwargs["flag"]} is unknown (see --help)')
        exit(0)

    # special flags mlflow & tensorboard
    if kwargs['flag'] in ['mlflow', 'tensorboard']:
        cd_dir = f'{join(os.environ.get("DATA_DIR"), "results")}'
        if kwargs['flag'] == 'mlflow':
            subprocess.run(f'cd {cd_dir}; mlflow ui', shell=True)
        elif kwargs['flag'] == 'tensorboard':
            subprocess.run(f'cd {cd_dir}; tensorboard --logdir tensorboard --reload_multifile=true', shell=True)

    # parse argument flag_args
    if len(kwargs['flag_args']):
        experiment_flags = [
            'show_experiment_configs',
            'show_experiment_config',
            'run_experiment',
            'get_experiment_results',
            'get_experiments_results',
            'predict',
        ]
        dataset_flags = [
            'set_up_dataset',
            'analyze_data',
        ]
        text_input_flags = [
            'predict',
        ]
        if kwargs['flag'] in experiment_flags:
            kwargs['experiment_name'] = kwargs['flag_args'][0]
        if kwargs['flag'] in dataset_flags:
            kwargs['dataset_name'] = kwargs['flag_args'][0]
        if kwargs['flag'] in text_input_flags:
            kwargs['text_input'] = ' '.join(kwargs['flag_args'][1:]) if len(kwargs['flag_args']) > 1 else None
    kwargs.pop('flag_args')

    # NerBlackBoxMain
    nerblackbox_main = NerBlackBoxMain(**kwargs)
    nerblackbox_main.main()
