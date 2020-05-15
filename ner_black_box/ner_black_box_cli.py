
import os
import subprocess
from os.path import join
import click
from ner_black_box.ner_black_box_main import NerBlackBoxMain


@click.command()
@click.argument('arg', nargs=-1, type=click.UNPROCESSED)
@click.option("--data_dir",        default='data',  help="relative path data directory")
@click.option("--init",                      "flag", help="flag", flag_value='init')
@click.option("--show_experiment_config",    "flag", help="flag", flag_value='show_experiment_config')
@click.option("--set_up_dataset",            "flag", help="flag", flag_value='set_up_dataset')
@click.option("--analyze_data",              "flag", help="flag", flag_value='analyze_data')
@click.option("--run_experiment",            "flag", help="flag", flag_value='run_experiment')
@click.option("--get_experiment_results",    "flag", help="flag", flag_value='get_experiment_results')
@click.option("--get_experiments",           "flag", help="flag", flag_value='get_experiments')
@click.option("--get_experiments_results",   "flag", help="flag", flag_value='get_experiments_results')
@click.option("--mlflow",                    "flag", help="flag", flag_value='mlflow')
@click.option("--tensorboard",               "flag", help="flag", flag_value='tensorboard')
@click.option("--with_tags",       default=None,    help="--set_up_dataset")
@click.option("--modify",          default=None,    help="--set_up_dataset")
@click.option("--val_fraction",    default=None,    help="--set_up_dataset")
@click.option("--verbose",         default=None,    help="--set_up_dataset")
@click.option("--run_name",        default=None,    help="--run_experiment")
@click.option("--device",          default=None,    help="--run_experiment")
@click.option("--fp16",            default=None,    help="--run_experiment")
def main(data_dir, **kwargs_optional):

    # environ
    base_dir = os.getcwd()
    os.environ['BASE_DIR'] = base_dir
    os.environ['DATA_DIR'] = join(base_dir, data_dir)
    # print('BASE_DIR = ', os.environ.get('BASE_DIR'))
    # print('DATA_DIR = ', os.environ.get('DATA_DIR'))

    # kwargs
    kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}

    # special flags mlflow & tensorboard
    if kwargs['flag'] in ['mlflow', 'tensorboard']:
        cd_dir = f'{join(os.environ.get("DATA_DIR"), "results")}'
        if kwargs['flag'] == 'mlflow':
            subprocess.run(f'cd {cd_dir}; mlflow ui', shell=True)
        elif kwargs['flag'] == 'tensorboard':
            subprocess.run(f'cd {cd_dir}; tensorboard --logdir tensorboard --reload_multifile=true', shell=True)

    # parse argument arg
    if len(kwargs['arg']):
        experiment_flags = [
            'show_experiment_config',
            'run_experiment',
            'get_experiment_results',
            'get_experiments_results',
        ]
        dataset_flags = [
            'set_up_dataset',
            'analyze_data',
        ]
        if kwargs['flag'] in experiment_flags:
            kwargs['experiment_name'] = kwargs['arg'][0]
        elif kwargs['flag'] in dataset_flags:
            kwargs['dataset_name'] = kwargs['arg'][0]
    kwargs.pop('arg')

    # NerBlackBoxMain
    ner_black_box_main = NerBlackBoxMain(**kwargs)
    ner_black_box_main.main()
