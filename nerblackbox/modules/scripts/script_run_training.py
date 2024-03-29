import argparse
import torch
import mlflow
import gc
from os.path import join, isdir

import logging
import warnings

from nerblackbox.modules.ner_training.single_run import execute_single_run
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.training_config.training import Training

logging.basicConfig(
    level=logging.WARNING
)  # basic setting that is mainly applied to mlflow's default logging
warnings.filterwarnings("ignore")


def main(params: argparse.Namespace, log_dirs: argparse.Namespace) -> None:
    """
    Args:
        params: w/ keys training_name, run_name, device, fp16, training_run_name_nr
        log_dirs: w/ keys mlflow, tensorboard

    Returns: -
    """
    assert_that_training_hasnt_been_run_before(params.training_name)

    training = Training(
        training_name=params.training_name,
        from_config=params.from_config,
        run_name=params.run_name,
        device=params.device,
        fp16=params.fp16,
    )

    with mlflow.start_run(run_name=params.training_name):
        for k, v in training.params_and_hparams["general"].items():
            mlflow.log_param(k, v)

        for run_name_nr in training.runs_name_nr:
            # params & hparams: dict -> namespace
            params = argparse.Namespace(**training.runs_params[run_name_nr])
            hparams = argparse.Namespace(**training.runs_hparams[run_name_nr])

            # execute single run
            execute_single_run(params, hparams, log_dirs, training=True)

            # clear gpu memory
            clear_gpu_memory(
                device=params.device, verbose=params.logging_level == "debug"
            )


def assert_that_training_hasnt_been_run_before(training_name: str) -> None:
    """
    Args:
        training_name: e.g. 'my_training'
    """
    training_directory = join(env_variable("DIR_CHECKPOINTS"), training_name)
    if isdir(training_directory):
        raise Exception(
            f"ERROR! training = {training_name} has been run before ({training_directory} exists)"
        )


def clear_gpu_memory(device, verbose: bool = False):
    """
    clear object from GPU memory
    ----------------------------
    :param device:  [torch device]
    :param verbose: [bool]
    :return: -
    """
    if device.type == "cuda":
        if verbose:
            print(torch.cuda.memory_summary(device=device))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            object_counter = 0
            for obj in gc.get_objects():
                try:
                    if obj.is_cuda:
                        object_counter += 1
                        del obj
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
        print(f"> cleared {object_counter} objects from GPU memory")

        if verbose:
            print(torch.cuda.memory_summary(device=device))


def _parse_args(_parser, _args):
    """
    :param _parser: [argparse ArgumentParser]
    :param _args:   [argparse arguments]
    :return _params:   [argparse.Namespace] attr: training_name, run_name, device, fp16
    :return _log_dirs: [argparse.Namespace] attr: mlflow, tensorboard
    """
    # parsing
    _params = None
    for group in _parser._action_groups:
        group_dict = {
            a.dest: getattr(_args, a.dest, None) for a in group._group_actions
        }
        if group.title == "args_general":
            group_dict["device"] = torch.device(
                "cuda"
                if torch.cuda.is_available() and group_dict["device"] == "gpu"
                else "cpu"
            )
            group_dict["fp16"] = (
                True
                if group_dict["fp16"]
                and group_dict["device"]
                and group_dict["device"].type == "cuda"
                else False
            )
            group_dict["from_config"] = bool(group_dict["from_config"])
            if group_dict["run_name"] is not None and len(group_dict["run_name"]) == 0:
                group_dict["run_name"] = None
            _params = argparse.Namespace(**group_dict)

    # log_dirs
    _log_dirs_dict = {
        "mlflow": env_variable("DIR_MLFLOW"),
        "tensorboard": env_variable("DIR_TENSORBOARD"),
        "checkpoints": env_variable("DIR_CHECKPOINTS"),
        "log_file": env_variable("LOG_FILE"),
        "mlflow_file": env_variable("MLFLOW_FILE"),
    }
    _log_dirs = argparse.Namespace(**_log_dirs_dict)

    return _params, _log_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # params
    args_general = parser.add_argument_group("args_general")
    args_general.add_argument(
        "--training_name", type=str, required=True
    )  # .. logging w/ mlflow & tensorboard
    args_general.add_argument(
        "--from_config", type=int, required=True
    )  # .. logging w/ mlflow & tensorboard
    args_general.add_argument(
        "--run_name", type=str, required=True
    )  # .. logging w/ mlflow & tensorboard
    args_general.add_argument("--device", type=str, required=True)  # .. device
    args_general.add_argument("--fp16", type=int, required=True)  # .. device

    args = parser.parse_args()
    _params, _log_dirs = _parse_args(parser, args)
    main(_params, _log_dirs)
