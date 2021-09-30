import pytest
from typing import Union, Dict, Optional, List

from pkg_resources import resource_filename
import os

from nerblackbox.modules.experiment_config.experiment_config import ExperimentConfig

os.environ["DATA_DIR"] = resource_filename(
    "nerblackbox", f"tests/test_data"
)


class TestExperimentConfig:

    experiment_config = ExperimentConfig(experiment_name="test_experiment",
                                         run_name="runA",
                                         device="gpu",
                                         fp16=True)

    def test_no_config_file(self):
        with pytest.raises(Exception):
            _ = ExperimentConfig(experiment_name="test_experiment_that_does_not_exist",
                                 run_name="runA",
                                 device="gpu",
                                 fp16=True)

    @pytest.mark.parametrize(
        "default, config_dict",
        [
            (
                    False,
                    {
                        "params": {
                            "dataset_name": "swedish_ner_corpus",
                            "annotation_scheme": "plain",
                            "prune_ratio_train": 0.01,
                            "prune_ratio_val": 0.01,
                            "prune_ratio_test": 0.01,
                            "train_on_val": True,
                            "train_on_test": True,
                            "pretrained_model_name": "KB/bert-base-swedish-cased",
                            "multiple_runs": 1,
                            "uncased": False,
                        },
                        "hparams": {
                            "max_epochs": 3,
                            "lr_max": 2e-5,
                            "lr_schedule": "hybrid",
                            "early_stopping": False,
                        },
                        "runA": {
                            "lr_cooldown_epochs": 1,
                        },
                    },
            ),
            (
                    True,
                    {
                        "params": {
                            "prune_ratio_train": 1.00,
                            "prune_ratio_val": 1.00,
                            "prune_ratio_test": 1.00,
                            "train_on_val": False,
                            "train_on_test": False,
                            "checkpoints": True,
                            "logging_level": "info",
                            "multiple_runs": 1,
                            "seed": 42,
                        },
                        "hparams": {
                            "batch_size": 16,
                            "max_seq_length": 64,
                            "max_epochs": 50,
                            "early_stopping": True,
                            "monitor": "val_loss",
                            "mode": "min",
                            "min_delta": 0.0,
                            "patience": 0,
                            "lr_warmup_epochs": 2,
                            "lr_cooldown_epochs": 3,
                            "lr_cooldown_restarts": False,
                            "lr_num_cycles": 4,
                        },
                    }
            ),
        ]
    )
    def test_get_config(self,
                        default: bool,
                        config_dict: Dict[str, Dict[str, str]]):
        _, test_config_dict = self.experiment_config._get_config(default=default)
        assert sorted(list(test_config_dict.keys())) == sorted(list(config_dict.keys())), \
            f"ERROR! test_config_dict.keys() = {sorted(list(test_config_dict.keys()))} does not equal " \
            f"config_dict.keys() = {sorted(list(config_dict.keys()))}"

        keys = list(test_config_dict.keys())
        for key in keys:
            assert sorted(test_config_dict[key].items()) == sorted(config_dict[key].items()), \
                f"ERROR! sorted(test_config_dict[{key}].items()) = {sorted(test_config_dict[key].items())} " \
                f"!= {sorted(config_dict[key].items())}"

    @pytest.mark.parametrize(
        "run_name_nr, params_and_hparams",
        [
            (
                    None,
                    {
                        # params
                        "dataset_name": "swedish_ner_corpus",  # update non-default
                        "annotation_scheme": "plain",  # update non-default
                        "prune_ratio_train": 0.01,  # update non-default
                        "prune_ratio_val": 0.01,  # update non-default
                        "prune_ratio_test": 0.01,  # update non-default
                        "train_on_val": True,  # update non-default
                        "train_on_test": True,  # update non-default
                        "checkpoints": True,
                        "logging_level": "info",
                        "multiple_runs": 1,
                        "seed": 42,
                        "pretrained_model_name": "KB/bert-base-swedish-cased",  # update non-default
                        "uncased": False,  # update non-default

                        # hparams
                        "batch_size": 16,
                        "max_seq_length": 64,
                        "max_epochs": 3,  # update non-default
                        "early_stopping": False,  # update non-default
                        "monitor": "val_loss",
                        "mode": "min",
                        "min_delta": 0.0,
                        "patience": 0,
                        "lr_warmup_epochs": 2,
                        "lr_cooldown_epochs": 3,
                        "lr_cooldown_restarts": False,
                        "lr_num_cycles": 4,
                        "lr_max": 2e-5,  # update non-default
                        "lr_schedule": "hybrid",  # update non-default
                    }
            ),
            (
                    "runA",
                    {
                        "lr_cooldown_epochs": 1,
                    }
            )
        ]
    )
    def test_get_params_and_hparams(self,
                                    run_name_nr: Optional[str],
                                    params_and_hparams: Dict[str, Union[str, int, float, bool]]):
        test_params_and_hparams = self.experiment_config.get_params_and_hparams(run_name_nr)
        assert sorted(list(test_params_and_hparams.keys())) == sorted(list(params_and_hparams.keys())), \
            f"ERROR! test_params_and_hparams.keys() = {sorted(list(test_params_and_hparams.keys()))} does not equal " \
            f"params_and_hparams.keys() = {sorted(list(params_and_hparams.keys()))}"

        keys = list(test_params_and_hparams.keys())
        for key in keys:
            assert test_params_and_hparams[key] == params_and_hparams[key], \
                f"ERROR! test_params_and_hparams[{key}] = {test_params_and_hparams[key]} " \
                f"!= {params_and_hparams[key]}"

    @pytest.mark.parametrize(
        "runs_name_nr, runs_params, runs_hparams",
        [
            (
                ["runA-1"],
                {
                    "runA-1": {
                        "experiment_name": "test_experiment",
                        "run_name": "runA",
                        "run_name_nr": "runA-1",
                        "device": "gpu",
                        "fp16": True,
                        "experiment_run_name_nr": "test_experiment/runA-1",

                        # params
                        "dataset_name": "swedish_ner_corpus",  # update non-default
                        "annotation_scheme": "plain",  # update non-default
                        "prune_ratio_train": 0.01,  # update non-default
                        "prune_ratio_val": 0.01,  # update non-default
                        "prune_ratio_test": 0.01,  # update non-default
                        "train_on_val": True,  # update non-default
                        "train_on_test": True,  # update non-default
                        "checkpoints": True,
                        "logging_level": "info",
                        "multiple_runs": 1,
                        "seed": 42,
                        "pretrained_model_name": "KB/bert-base-swedish-cased",  # update non-default
                        "uncased": False,  # update non-default
                    }
                },
                {
                    "runA-1": {
                        # hparams
                        "batch_size": 16,
                        "max_seq_length": 64,
                        "max_epochs": 3,  # update non-default
                        "early_stopping": False,  # update non-default
                        "monitor": "val_loss",
                        "mode": "min",
                        "min_delta": 0.0,
                        "patience": 0,
                        "lr_warmup_epochs": 2,
                        # "lr_cooldown_epochs": 3,
                        "lr_cooldown_restarts": False,
                        "lr_num_cycles": 4,
                        "lr_max": 2e-5,  # update non-default
                        "lr_schedule": "hybrid",  # update non-default

                        "lr_cooldown_epochs": 1,  # run-specific
                    }
                },
            )
        ]
    )
    def test_parse(self,
                   runs_name_nr: List[str],
                   runs_params: Dict[str, Dict[str, Union[str, int, float, bool]]],
                   runs_hparams: Dict[str, Dict[str, Union[str, int, float, bool]]]):
        test_runs_name_nr, test_runs_params, test_runs_hparams = self.experiment_config.parse()

        # 1. runs_name_nr
        assert test_runs_name_nr == runs_name_nr, f"ERROR! test_runs_name_nr = {test_runs_name_nr} != {runs_name_nr}"

        # 2. runs_params
        assert sorted(list(test_runs_params.keys())) == sorted(list(runs_params.keys())), \
            f"ERROR! test_runs_params.keys() = {sorted(list(test_runs_params.keys()))} does not equal " \
            f"runs_params.keys() = {sorted(list(runs_params.keys()))}"
        keys = list(test_runs_params.keys())
        for key in keys:
            assert test_runs_params[key] == runs_params[key], \
                f"ERROR! test_runs_params[{key}] = {test_runs_params[key]} " \
                f"!= {runs_params[key]}"

    @pytest.mark.parametrize(
        "input_key, input_value, converted_input, error",
        [
            (
                    "lr_schedule",
                    "constant",
                    "constant",
                    False,
            ),
            (
                    "prune_ratio_train",
                    "0.01",
                    0.01,
                    False,
            ),
            (
                    "checkpoints",
                    "False",
                    False,
                    False,
            ),
            (
                    "multiple_runs",
                    "5",
                    5,
                    False,
            ),
            (
                    "xyz",
                    "",
                    "",
                    True,
            ),
        ]
    )
    def test_convert(self,
                     input_key: str,
                     input_value: str,
                     converted_input: Union[str, int, float, bool],
                     error: bool):
        if error:
            with pytest.raises(Exception):
                _ = self.experiment_config._convert(input_key, input_value)
        else:
            test_converted_input = self.experiment_config._convert(input_key, input_value)
            assert test_converted_input == converted_input, \
                f"ERROR! test_converted_input = {test_converted_input} != {converted_input} " \
                f"for input_key = {input_key} and input_value = {input_value}"
