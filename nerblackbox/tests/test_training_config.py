import pytest
from typing import Union, Dict, Optional, List

from pkg_resources import resource_filename
import os

from nerblackbox.modules.training_config.training_config import TrainingConfig

os.environ["DATA_DIR"] = resource_filename("nerblackbox", f"tests/test_data")


class TestTrainingConfig:

    training_config_default = TrainingConfig(training_name="default")
    training_config = TrainingConfig(training_name="test_training")

    # 1 ################################################################################################################
    def test_no_config_file(self):
        with pytest.raises(Exception):
            _ = TrainingConfig(training_name="test_training_that_does_not_exist")

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
                        "train_fraction": 0.01,
                        "val_fraction": 0.01,
                        "test_fraction": 0.01,
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
                        "train_fraction": 1.00,
                        "val_fraction": 1.00,
                        "test_fraction": 1.00,
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
            test_config_dict = self.training_config_default.config
        else:
            test_config_dict = self.training_config.config
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
                "train_fraction",
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
                _ = self.training_config._convert(input_key, input_value)
        else:
            test_converted_input = self.training_config._convert(
                input_key, input_value
            )
            assert test_converted_input == converted_input, (
                f"ERROR! test_converted_input = {test_converted_input} != {converted_input} "
                f"for input_key = {input_key} and input_value = {input_value}"
            )
