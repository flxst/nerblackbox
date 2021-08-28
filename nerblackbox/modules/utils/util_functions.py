import os
from typing import Tuple, List, Optional
import numpy as np
from os.path import join
from argparse import Namespace
import pkg_resources
from omegaconf import OmegaConf

from nerblackbox.modules.utils.env_variable import env_variable


def get_available_datasets():
    """
    get datasets that are available in DIR_DATASETS directory
    ---------------------------------------------------------
    :return: available datasets: [list] of [str], e.g. ['suc', 'swedish_ner_corpus']
    """
    dir_datasets = env_variable("DIR_DATASETS")
    return [
        folder
        for folder in os.listdir(dir_datasets)
        if os.path.isdir(join(dir_datasets, folder))
    ]


def get_dataset_path(dataset):
    """
    get dataset path for dataset
    ----------------------------
    :param dataset:        [str] dataset name, e.g. 'suc', 'swedish_ner_corpus'
    :return: dataset_path: [str] path to dataset directory
    """
    return join(env_variable("DIR_DATASETS"), dataset)


def get_hardcoded_parameters(keys=False):
    """
    :param keys: [bool] whether to return [list] of keys instead of whole [dict]
    :return: _general:  [dict] w/ keys = parameter name [str] & values = type [str] --- or [list] of keys
    :return: _params:   [dict] w/ keys = parameter name [str] & values = type [str] --- or [list] of keys
    :return: _hparams:  [dict] w/ keys = parameter name [str] & values = type [str] --- or [list] of keys
    :return: _log_dirs: [dict] w/ keys = parameter name [str] & values = type [str] --- or [list] of keys
    """
    _general = {
        "experiment_name": "str",
        "run_name": "str",
        "run_name_nr": "str",
        "device": "str",
        "fp16": "bool",
        "experiment_run_name_nr": "str",
    }
    _params = {
        "dataset_name": "str",
        "annotation_scheme": "str",
        "prune_ratio_train": "float",
        "prune_ratio_val": "float",
        "prune_ratio_test": "float",
        "pretrained_model_name": "str",
        "uncased": "bool",
        "checkpoints": "bool",
        "logging_level": "str",
        "multiple_runs": "int",
        "seed": "int",
    }
    _hparams = {
        "batch_size": "int",
        "max_seq_length": "int",
        "max_epochs": "int",
        "early_stopping": "bool",
        "monitor": "str",
        "min_delta": "float",
        "patience": "int",
        "mode": "str",
        "lr_max": "float",
        "lr_schedule": "str",
        "lr_warmup_epochs": "int",
        "lr_cooldown_epochs": "int",
        "lr_cooldown_restarts": "bool",
        "lr_num_cycles": "int",
    }
    _log_dirs = {
        "mlflow": "str",
        "tensorboard": "str",
        "checkpoints": "str",
        "log_file": "str",
        "mlflow_file": "str",
    }
    if keys:
        return (
            list(_general.keys()),
            list(_params.keys()),
            list(_hparams.keys()),
            list(_log_dirs.keys()),
        )
    else:
        return _general, _params, _hparams, _log_dirs


def unify_parameters(_params, _hparams, _log_dirs, _experiment):
    """
    unify parameters (namespaces, bool) to one namespace
    ----------------------------------------------------
    :param _params:             [Namespace] with keys = 'dataset_name', 'annotation_scheme', ..
    :param _hparams:            [Namespace] with keys = 'batch_size', 'max_seq_length', ..
    :param _log_dirs:           [Namespace] with keys = 'mlflow', 'tensorboard', ..
    :param _experiment:         [bool]
    :return: _lightning_hparams [OmegaConf] with keys = all keys from input namespaces + 'experiment'
    """
    _dict = dict()
    _dict.update(vars(_params))
    _dict.update(vars(_hparams))
    _dict.update(vars(_log_dirs))
    _dict.update({"experiment": _experiment})
    _lightning_hparams = Namespace(**_dict)
    _lightning_hparams.device = (
        _lightning_hparams.device.type
    )  # needs to be a string (not torch.device) for logging

    return OmegaConf.create(vars(_lightning_hparams))


def split_parameters(_lightning_hparams):
    """
    split namespace to parameters (namespaces, bool)
    ----------------------------------------------------
    :param _lightning_hparams [Namespace] with keys = all keys from output namespaces + 'experiment'
    :return: _params:         [Namespace] with keys = 'dataset_name', 'annotation_scheme', ..
    :return: _hparams:        [Namespace] with keys = 'batch_size', 'max_seq_length', ..
    :return: _log_dirs:       [Namespace] with keys = 'mlflow', 'tensorboard', ..
    :return: _experiment:     [bool]
    """
    keys_general, keys_params, keys_hparams, keys_log_dirs = get_hardcoded_parameters(
        keys=True
    )
    _params = Namespace(
        **{
            k: v
            for k, v in _lightning_hparams.items()
            if k in keys_general + keys_params
        }
    )
    _hparams = Namespace(
        **{k: v for k, v in _lightning_hparams.items() if k in keys_hparams}
    )
    _log_dirs = Namespace(
        **{k: v for k, v in _lightning_hparams.items() if k in keys_log_dirs}
    )
    _experiment = _lightning_hparams.get("experiment")
    return _params, _hparams, _log_dirs, _experiment


def get_package_version():
    return pkg_resources.get_distribution("nerblackbox").version


def checkpoint2epoch(_checkpoint_name):
    """
    :param _checkpoint_name: [str], e.g. 'epoch=2.ckpt' or 'epoch=2_v0.ckpt'
    :return: _epoch:         [int], e.g. 2
    """
    return int(_checkpoint_name.split("epoch=")[-1].split("_")[0].replace(".ckpt", ""))


def epoch2checkpoint(_epoch):
    """
    :param _epoch:            [int], e.g. 2
    :return _checkpoint_name: [str], e.g. 'epoch=2.ckpt' or 'epoch=2_v0.ckpt'
    """
    return f"epoch={_epoch}.ckpt"


def get_run_name(_run_name_nr):
    """
    :param _run_name_nr: [str], e.g. 'runA-1'
    :return: _run_name:  [str], e.g. 'runA'
    """
    return _run_name_nr.split("-")[0]


def get_run_name_nr(_run_name, _run_nr):
    """
    :param _run_name:      [str], e.g. 'runA'
    :param _run_nr:        [int], e.g. 1
    :return: _run_name_nr: [str], e.g. 'runA-1'
    """
    return f"{_run_name}-{_run_nr}"


def compute_mean_and_dmean(values: List[float]) -> Tuple[float, Optional[float], str]:
    """
    compute mean and its error dmean = std deviation / sqrt(N)
    ----------------------------------------------------------
    :param values: [list / np array] of [float]
    :return: mean  [float]
    :return: dmean [float]
    :return: convergence_str [str] e.g. "3/5" (i.e. 3 of 5 runs converged)
    """
    if len(values) == 0:
        raise Exception(f"cannot compute mean and dmean of empty list!")
    else:
        convergence_runs = int(sum([1 for value in values if value != -1]))
        convergence_values = [value for value in values if value != -1]
        convergence_str = f"{convergence_runs}/{len(values)}"
        if len(convergence_values) == 0:
            return -1, None, convergence_str
        if len(values) == 1:
            return convergence_values[0], None, convergence_str
        elif len(values) > 1:
            return float(np.mean(convergence_values)), \
                   float(np.std(convergence_values) / np.sqrt(len(convergence_values))), \
                   convergence_str
