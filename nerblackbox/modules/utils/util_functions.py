import os
from typing import Tuple, List, Optional, Union
import numpy as np
from os.path import join
from argparse import Namespace
import pkg_resources
from omegaconf import OmegaConf, DictConfig, ListConfig

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


def unify_parameters(_params: Namespace,
                     _hparams: Namespace,
                     _log_dirs: Namespace,
                     _experiment: bool) -> Union[DictConfig, ListConfig]:
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

    return OmegaConf.create(vars(_lightning_hparams))


def split_parameters(_lightning_hparams: Namespace) -> Tuple[Namespace, Namespace, Namespace, bool]:
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
    _params = Namespace(
        **{
            k: v
            for k, v in _lightning_hparams.items()
            if k in list(GENERAL.keys()) + list(PARAMS.keys())
        }
    )
    _hparams = Namespace(
        **{k: v for k, v in _lightning_hparams.items() if k in list(HPARAMS.keys())}
    )
    _log_dirs = Namespace(
        **{k: v for k, v in _lightning_hparams.items() if k in list(LOG_DIRS.keys())}
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
    elif len(values) > 1:
        return float(np.mean(values)), float(np.std(values) / np.sqrt(len(values)))
