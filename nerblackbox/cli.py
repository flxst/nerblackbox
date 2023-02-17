"""Command Line Interface of the nerblackbox package."""

import os
import subprocess
from os.path import join
import click
from nerblackbox.api.store import Store
from nerblackbox.api.dataset import Dataset
from nerblackbox.api.experiment import Experiment
from nerblackbox.api.model import Model
from nerblackbox.modules.experiment_results import ExperimentResults
from nerblackbox.modules.utils.env_variable import env_variable


########################################################################################################################
# CLI
########################################################################################################################
@click.group()
@click.option(
    "--store_dir", default="store", type=str, help="[str] relative path of store directory"
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
    data_dir = kwargs.pop("store_dir")
    if len(data_dir):
        Store.set_path(data_dir)

    # context
    ctx.obj = kwargs


########################################################################################################################
# COMMANDS
########################################################################################################################
@nerbb.command(name="analyze_data")
@click.argument("dataset_name")
def analyze_data(dataset_name: str):
    """analyze a dataset."""
    dataset = Dataset(dataset_name)
    dataset.overview()


@click.pass_context
@nerbb.command(name="clear_data")
def clear_data(ctx):
    """clear data (checkpoints and optionally results)."""
    context = {k: v for k, v in ctx.obj.items() if k in ["results"]}
    Store.clear_data(**context)


@nerbb.command(name="create")
def create():
    """
    initialize the data_dir directory.
    needs to be called exactly once before any other CLI/API commands of the package are executed.
    """
    Store.create()


@nerbb.command(name="show_experiments")
def show_experiments():
    """get overview on experiments."""
    Store.show_experiments()


@nerbb.command(name="get_experiment_results")
@click.argument("experiment_name")
def get_experiment_results(experiment_name: str):
    """get results for a single experiment."""
    experiment = Experiment(experiment_name)
    assert isinstance(experiment.results, ExperimentResults), \
        f"ERROR! experiment.results is not an instance of ExperimentResults."
    for attribute in ["best_single_run"]:
        assert hasattr(experiment.results, attribute), \
            f"ERROR! experiment.results does not have attribute = {attribute}"

    for average in [False, True]:
        score = experiment.get_result(metric="f1", level="entity", label="micro", phase="test", average=average)
        print(f"score (average={average}) = {score}")
        assert isinstance(score, str), \
            f"ERROR! experiment.get_result() did not return a str for average = {average}."


@nerbb.command(name="mlflow")
def mlflow():
    """show detailed experiment results in mlflow (port = 5000)."""
    cd_dir = f'{join(env_variable("DATA_DIR"), "results")}'
    subprocess.run(f"cd {cd_dir}; mlflow ui", shell=True)


@nerbb.command(name="predict")
@click.argument("experiment_name")
@click.argument("text_input")
def predict(experiment_name: str, text_input: str):
    """predict labels for text_input using the best model of a single experiment."""
    model = Model.from_experiment(experiment_name)
    assert isinstance(model, Model), f"ERROR! model from experiment {experiment_name} could not be loaded."
    predictions = model.predict(text_input)
    print(predictions)


@nerbb.command(name="predict_proba")
@click.argument("experiment_name")
@click.argument("text_input")
def predict_proba(experiment_name: str, text_input: str):
    """predict label probabilities for text_input using the best model of a single experiment."""
    model = Model.from_experiment(experiment_name)
    assert isinstance(model, Model), f"ERROR! model from experiment {experiment_name} could not be loaded."
    predictions = model.predict_proba(text_input)
    print(predictions)


@nerbb.command(name="run_experiment")
@click.pass_context
@click.argument("experiment_name")
def run_experiment(ctx, experiment_name: str):
    """run a single experiment."""
    context = {k: v for k, v in ctx.obj.items() if k in ["run_name", "device", "fp16"]}
    experiment = Experiment(experiment_name, from_config=True, **context)
    experiment.run()


@nerbb.command(name="set_up_dataset")
@click.pass_context
@click.argument("dataset_and_subset_name")
def set_up_dataset(ctx, dataset_and_subset_name: str):
    """set up a dataset using the associated Formatter class."""
    split_list = dataset_and_subset_name.split()
    dataset = Dataset(
        dataset_name=split_list[0],
        dataset_subset_name=split_list[-1] if len(split_list) > 1 else "",
    )
    context = {k: v for k, v in ctx.obj.items() if k in ["modify", "val_fraction", "verbose"]}
    dataset.set_up(**context)


@nerbb.command(name="show_experiment_config")
@click.argument("experiment_name")
def show_experiment_config(experiment_name: str):
    """show a single experiment configuration in detail."""
    experiment = Experiment(experiment_name)
    experiment.show_config()


@nerbb.command(name="tensorboard")
def tensorboard():
    """show detailed experiment results in tensorboard. (port = 6006)."""
    cd_dir = f'{join(env_variable("DATA_DIR"), "results")}'
    subprocess.run(
        f"cd {cd_dir}; tensorboard --logdir tensorboard --reload_multifile=true",
        shell=True,
    )
