"""Command Line Interface of the nerblackbox package."""

import subprocess
from os.path import join
import click
from nerblackbox.api.store import Store
from nerblackbox.modules.utils.env_variable import env_variable


########################################################################################################################
# CLI
########################################################################################################################
@click.group()
@click.option(
    "--store_dir",
    default="store",
    type=str,
    help="[str] relative path of store directory",
)
@click.pass_context
def nerblackbox(ctx, **kwargs_optional):
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
@nerblackbox.command(name="mlflow")
def mlflow():
    """show detailed experiment results in mlflow (port = 5000)."""
    cd_dir = f'{join(env_variable("DATA_DIR"), "results")}'
    subprocess.run(f"cd {cd_dir}; mlflow ui", shell=True)


@nerblackbox.command(name="tensorboard")
def tensorboard():
    """show detailed experiment results in tensorboard. (port = 6006)."""
    cd_dir = f'{join(env_variable("DATA_DIR"), "results")}'
    subprocess.run(
        f"cd {cd_dir}; tensorboard --logdir tensorboard --reload_multifile=true",
        shell=True,
    )
