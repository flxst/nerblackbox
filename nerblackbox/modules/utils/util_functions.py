import os
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from os.path import join, isfile
import json
from argparse import Namespace
import pkg_resources
from omegaconf import OmegaConf, DictConfig

from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.utils.parameters import GENERAL, PARAMS, HPARAMS, LOG_DIRS


def get_available_datasets() -> List[str]:
    """
    get datasets that are available in DIR_DATASETS directory

    Returns:
        available datasets: e.g. ['suc', 'swedish_ner_corpus']
    """
    dir_datasets = env_variable("DIR_DATASETS")
    return [
        folder
        for folder in os.listdir(dir_datasets)
        if os.path.isdir(join(dir_datasets, folder))
    ]


def get_dataset_path(dataset: str) -> str:
    """
    get dataset path for dataset

    Args:
        dataset: e.g. 'suc', 'swedish_ner_corpus'

    Returns:
        dataset_path: path to dataset directory
    """
    return join(env_variable("DIR_DATASETS"), dataset)


def read_special_tokens(dataset: str) -> List[str]:
    """
    if the file "special_tokens.json" exists for the dataset, read it and return its content
    ----------------------------------------------------------------------------------------

    Args:
        dataset: e.g. 'swedish_ner_corpus'

    Returns:
        special_tokens: e.g. ["[NEWLINE]", "[TAB]"]
    """
    path_special_tokens = join(get_dataset_path(dataset), "special_tokens.json")
    if isfile(path_special_tokens):
        with open(path_special_tokens, "r") as file:
            special_tokens = json.load(file)
        assert isinstance(
            special_tokens, list
        ), f"ERROR! {path_special_tokens} seems to contain a {type(special_tokens)}, should be list."
    else:
        special_tokens = list()
    return special_tokens


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


def unify_parameters(
    _params: Namespace, _hparams: Namespace, _log_dirs: Namespace, _experiment: bool
) -> DictConfig:
    """
    unify parameters (namespaces, bool) to one namespace

    Args:
        _params:        keys = 'dataset_name', 'annotation_scheme', ..
        _hparams:       keys = 'batch_size', 'max_seq_length', ..
        _log_dirs:      keys = 'mlflow', 'tensorboard', ..
        _experiment:

    Returns:
        _lightning_hparams: keys = all keys from input namespaces + 'experiment'
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

    omega_conf = OmegaConf.create(vars(_lightning_hparams))
    assert (
        type(omega_conf) == DictConfig
    ), f"ERROR! type(omega_conf) = {type(omega_conf)} should be DictConfig"
    return omega_conf


def split_parameters(
    _lightning_hparams: DictConfig,
) -> Tuple[Namespace, Namespace, Namespace, bool]:
    """
    split namespace to parameters (namespaces, bool)

    Args:
        _lightning_hparams: keys = all keys from output namespaces + 'experiment'

    Returns:
        _params:            keys = 'dataset_name', 'annotation_scheme', ..
        _hparams:           keys = 'batch_size', 'max_seq_length', ..
        _log_dirs:          keys = 'mlflow', 'tensorboard', ..
        _experiment:
    """
    _lightning_hparams_dict = OmegaConf.to_container(_lightning_hparams)
    assert isinstance(
        _lightning_hparams_dict, dict
    ), f"ERROR! {type(_lightning_hparams_dict)}"

    _params = Namespace(
        **{
            str(k): v
            for k, v in _lightning_hparams_dict.items()
            if k in list(GENERAL.keys()) + list(PARAMS.keys())
        }
    )
    _hparams = Namespace(
        **{
            str(k): v
            for k, v in _lightning_hparams_dict.items()
            if k in list(HPARAMS.keys())
        }
    )
    _log_dirs = Namespace(
        **{
            str(k): v
            for k, v in _lightning_hparams_dict.items()
            if k in list(LOG_DIRS.keys())
        }
    )
    _experiment = _lightning_hparams.get("experiment")
    return _params, _hparams, _log_dirs, _experiment


def get_package_version() -> str:
    return pkg_resources.get_distribution("nerblackbox").version


def checkpoint2epoch(_checkpoint_name: str) -> int:
    """
    Args:
        _checkpoint_name: e.g. 'epoch=2.ckpt' or 'epoch=2_v0.ckpt'

    Returns:
        _epoch:           e.g. 2
    """
    return int(_checkpoint_name.split("epoch=")[-1].split("_")[0].replace(".ckpt", ""))


def epoch2checkpoint(_epoch: int) -> str:
    """
    Args:
        _epoch:           e.g. 2

    Returns:
        _checkpoint_name: e.g. 'epoch=2.ckpt'
    """
    return f"epoch={_epoch}.ckpt"


def get_run_name(_run_name_nr: str) -> str:
    """
    Args:
        _run_name_nr: e.g. 'runA-1'

    Returns:
        _run_name:    e.g. 'runA'
    """
    return _run_name_nr.split("-")[0]


def get_run_name_nr(_run_name: str, _run_nr: int) -> str:
    """
    Args:
        _run_name: e.g. 'runA'
        _run_nr:   e.g. 1

    Returns:
        _run_name_nr: e.g. 'runA-1'
    """
    return f"{_run_name}-{_run_nr}"


def compute_mean_and_dmean(values: List[float]) -> Tuple[float, Optional[float]]:
    """
    compute mean and its error dmean = std deviation / sqrt(N)

    Args:
        values: e.g. [1., 2.]

    Returns:
        mean:   e.g. 1.5
        dmean:  e.g. 0.35355 = 0.5 / sqrt(2)
    """
    if len(values) == 0:
        return -1, None
    elif len(values) == 1:
        return values[0], None
    else:  # i.e. if len(values) > 1:
        return float(np.mean(values)), float(np.std(values) / np.sqrt(len(values)))
