
import os
from os.path import join
import click
import ner_black_box.ner_black_box_main as ner_black_box_main


@click.command()
@click.option("--data_dir",        default='data',  help="relative path data directory")
@click.option("--init",            "flag", help="flag", flag_value='init')
@click.option("--set_up_dataset",  "flag", help="flag", flag_value='set_up_dataset')
@click.option("--analyze_data",    "flag", help="flag", flag_value='analyze_data')
@click.option("--run_experiment",  "flag", help="flag", flag_value='run_experiment')
@click.option("--dataset",         default=None,    help="--set_up_dataset")
@click.option("--with_tags",       default=None,    help="--set_up_dataset")
@click.option("--modify",          default=None,    help="--set_up_dataset")
@click.option("--val_fraction",    default=None,    help="--set_up_dataset")
@click.option("--verbose",         default=None,    help="--set_up_dataset")
@click.option("--experiment_name", default=None,    help="--run_experiment")
@click.option("--run_name",        default=None,    help="--run_experiment")
@click.option("--device",          default=None,    help="--run_experiment")
@click.option("--fp16",            default=None,    help="--run_experiment")
def main(data_dir, **kwargs_optional):

    kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}

    base_dir = os.getcwd()
    os.environ['BASE_DIR'] = base_dir
    os.environ['DATA_DIR'] = join(base_dir, data_dir)
    print('BASE_DIR = ', os.environ.get('BASE_DIR'))
    print('DATA_DIR = ', os.environ.get('DATA_DIR'))

    ner_black_box_main.main(**kwargs)
