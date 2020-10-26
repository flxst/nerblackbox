import os
import numpy as np
from os.path import join
from argparse import Namespace
import pkg_resources

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
        "dataset_tags": "str",
        "prune_ratio_train": "float",
        "prune_ratio_val": "float",
        "prune_ratio_test": "float",
        "pretrained_model_name": "str",
        "uncased": "bool",
        "checkpoints": "bool",
        "logging_level": "str",
        "multiple_runs": "int",
    }
    _hparams = {
        "batch_size": "int",
        "max_seq_length": "int",
        "max_epochs": "int",
        "monitor": "str",
        "min_delta": "float",
        "patience": "int",
        "mode": "str",
        "lr_max": "float",
        "lr_schedule": "str",
        "lr_warmup_epochs": "int",
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
    :param _params:             [Namespace] with keys = 'dataset_name', 'dataset_tags', ..
    :param _hparams:            [Namespace] with keys = 'batch_size', 'max_seq_length', ..
    :param _log_dirs:           [Namespace] with keys = 'mlflow', 'tensorboard', ..
    :param _experiment:         [bool]
    :return: _lightning_hparams [Namespace] with keys = all keys from input namespaces + 'experiment'
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
    return _lightning_hparams


def split_parameters(_lightning_hparams):
    """
    split namespace to parameters (namespaces, bool)
    ----------------------------------------------------
    :param _lightning_hparams [Namespace] with keys = all keys from output namespaces + 'experiment'
    :return: _params:         [Namespace] with keys = 'dataset_name', 'dataset_tags', ..
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
            for k, v in vars(_lightning_hparams).items()
            if k in keys_general + keys_params
        }
    )
    _hparams = Namespace(
        **{k: v for k, v in vars(_lightning_hparams).items() if k in keys_hparams}
    )
    _log_dirs = Namespace(
        **{k: v for k, v in vars(_lightning_hparams).items() if k in keys_log_dirs}
    )
    _experiment = vars(_lightning_hparams).get("experiment")
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


def compute_mean_and_dmean(values):
    """
    compute mean and its error dmean = std deviation / sqrt(N)
    ----------------------------------------------------------
    :param values: [list / np array] of [float]
    :return: mean  [float]
    :return: dmean [float]
    """
    if len(values) == 1:
        return values[0], None
    elif len(values) > 1:
        return np.mean(values), np.std(values) / np.sqrt(len(values))
    else:
        raise Exception(f"cannot compute mean and dmean of empty list!")
