import pytest
from typing import Union, Dict, Optional, List

from pkg_resources import resource_filename
import os

from nerblackbox.modules.experiment_config.experiment_config import ExperimentConfig

os.environ["DATA_DIR"] = resource_filename("nerblackbox", f"tests/test_data")


class TestExperimentConfig:

    experiment_config_default = ExperimentConfig(experiment_name="default")
    experiment_config = ExperimentConfig(experiment_name="test_experiment")

    # 1 ################################################################################################################
    def test_no_config_file(self):
        with pytest.raises(Exception):
            _ = ExperimentConfig(experiment_name="test_experiment_that_does_not_exist")

    # 2 ################################################################################################################
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
                },
            ),
        ],
    )
    def test_get_config(self, default: bool, config_dict: Dict[str, Dict[str, str]]):
        if default:
            test_config_dict = self.experiment_config_default.config
        else:
            test_config_dict = self.experiment_config.config
        assert sorted(list(test_config_dict.keys())) == sorted(
            list(config_dict.keys())
        ), (
            f"ERROR! test_config_dict.keys() = {sorted(list(test_config_dict.keys()))} does not equal "
            f"config_dict.keys() = {sorted(list(config_dict.keys()))}"
        )

        keys = list(test_config_dict.keys())
        for key in keys:
            assert sorted(test_config_dict[key].items()) == sorted(
                config_dict[key].items()
            ), (
                f"ERROR! sorted(test_config_dict[{key}].items()) = {sorted(test_config_dict[key].items())} "
                f"!= {sorted(config_dict[key].items())}"
            )

    # 3 ################################################################################################################
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
        ],
    )
    def test_convert(
        self,
        input_key: str,
        input_value: str,
        converted_input: Union[str, int, float, bool],
        error: bool,
    ):
        if error:
            with pytest.raises(Exception):
                _ = self.experiment_config._convert(input_key, input_value)
        else:
            test_converted_input = self.experiment_config._convert(
                input_key, input_value
            )
            assert test_converted_input == converted_input, (
                f"ERROR! test_converted_input = {test_converted_input} != {converted_input} "
                f"for input_key = {input_key} and input_value = {input_value}"
            )
