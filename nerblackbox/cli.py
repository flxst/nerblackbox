"""Command Line Interface of the nerblackbox package."""

import os
import subprocess
from os.path import join
import click
from typing import Dict, Any
from nerblackbox.modules.main import NerBlackBoxMain


########################################################################################################################
# CLI
########################################################################################################################
@click.group()
@click.option(
    "--data_dir", default="data", type=str, help="[str] relative path of data directory"
)
@click.option(
    "--modify/--no-modify", default=False, help="[bool] if flag=set_up_dataset"
)
@click.option(
    "--val_fraction", default=None, type=float, help="[float] if flag=set_up_dataset"
)
@click.option(
    "--verbose/--no-verbose", default=False, help="[bool] if flag=set_up_dataset"
)
@click.option("--run_name", default=None, type=str, help="[str] if flag=run_experiment")
@click.option("--device", default=None, type=str, help="[str] if flag=run_experiment")
@click.option("--fp16/--no-fp16", default=False, help="[bool] if flag=run_experiment")
@click.option("--results/--no-results", default=False, help="[bool] if flag=clear_data")
@click.pass_context
def nerbb(ctx, **kwargs_optional):
    ctx.ensure_object(dict)

    # kwargs
    kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}

    # environ
    base_dir = os.getcwd()
    data_dir = kwargs.pop("data_dir")
    os.environ["BASE_DIR"] = base_dir
    os.environ["DATA_DIR"] = join(base_dir, data_dir)
    # print('BASE_DIR = ', os.environ.get('BASE_DIR'))
    # print('DATA_DIR = ', os.environ.get('DATA_DIR'))

    # context
    ctx.obj = kwargs


########################################################################################################################
# COMMANDS HELPER FUNCTION
########################################################################################################################
def _run_nerblackbox_main(_ctx_obj: Dict[str, Any], _kwargs: Dict[str, str]) -> None:
    """
    given context (_ctx_obj) and all relevant arguments (_kwargs), invoke NerBlackBoxMain
    is used by every nerbb command
    """
    kwargs = dict(**_ctx_obj, **_kwargs)

    nerblackbox_main = NerBlackBoxMain(**kwargs)
    nerblackbox_main.main()


########################################################################################################################
# COMMANDS
########################################################################################################################
@nerbb.command(name="analyze_data")
@click.pass_context
@click.argument("dataset_name")
def analyze_data(ctx, dataset_name: str):
    """analyze a dataset."""
    kwargs = {
        "flag": "analyze_data",
        "dataset_name": dataset_name,
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="clear_data")
@click.pass_context
def clear_data(ctx):
    """clear data (checkpoints and optionally results)."""
    kwargs = {
        "flag": "clear_data",
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="download")
@click.pass_context
def download(ctx):
    """
    download & prepare built-in datasets, prepare experiment configuration.
    needs to be called exactly once before any other CLI/API commands of the package are executed
    in case built-in datasets shall be used.
    """
    kwargs = {
        "flag": "download",
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="get_experiments")
@click.pass_context
def get_experiments(ctx):
    """get overview on experiments."""
    kwargs = {
        "flag": "get_experiments",
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="get_experiment_results")
@click.pass_context
@click.argument("experiment_name")
def get_experiment_results(ctx, experiment_name: str):
    """get results for a single experiment."""
    kwargs = {
        "flag": "get_experiment_results",
        "experiment_name": experiment_name,
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="init")
@click.pass_context
def init(ctx):
    """
    initialize the data_dir directory.
    needs to be called exactly once before any other CLI/API commands of the package are executed.
    """
    kwargs = {
        "flag": "init",
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="mlflow")
def mlflow():
    """show detailed experiment results in mlflow (port = 5000)."""
    cd_dir = f'{join(os.environ.get("DATA_DIR"), "results")}'
    subprocess.run(f"cd {cd_dir}; mlflow ui", shell=True)


@nerbb.command(name="predict")
@click.pass_context
@click.argument("experiment_name")
@click.argument("text_input")
def predict(ctx, experiment_name: str, text_input: str):
    """predict labels for text_input using the best model of a single experiment."""
    kwargs = {
        "flag": "predict",
        "experiment_name": experiment_name,
        "text_input": text_input,
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="run_experiment")
@click.pass_context
@click.argument("experiment_name")
def run_experiment(ctx, experiment_name: str):
    """run a single experiment."""
    kwargs = {
        "flag": "run_experiment",
        "experiment_name": experiment_name,
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="set_up_dataset")
@click.pass_context
@click.argument("dataset_name")
def set_up_dataset(ctx, dataset_name: str):
    """set up a dataset using the associated Formatter class."""
    kwargs = {
        "flag": "set_up_dataset",
        "dataset_name": dataset_name,
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="show_experiment_config")
@click.pass_context
@click.argument("experiment_name")
def show_experiment_config(ctx, experiment_name: str):
    """show a single experiment configuration in detail."""
    kwargs = {
        "flag": "show_experiment_config",
        "experiment_name": experiment_name,
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="show_experiment_configs")
@click.pass_context
def show_experiment_configs(ctx):
    """show overview on all available experiment configurations."""
    kwargs = {
        "flag": "show_experiment_configs",
    }
    _run_nerblackbox_main(ctx.obj, kwargs)


@nerbb.command(name="tensorboard")
def tensorboard():
    """show detailed experiment results in tensorboard. (port = 6006)."""
    cd_dir = f'{join(os.environ.get("DATA_DIR"), "results")}'
    subprocess.run(
        f"cd {cd_dir}; tensorboard --logdir tensorboard --reload_multifile=true",
        shell=True,
    )
