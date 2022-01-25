import pytest
from typing import Union, Dict, Optional, List

from pkg_resources import resource_filename
import os

from nerblackbox.modules.experiment_config.experiment import Experiment
from nerblackbox.modules.utils.util_functions import get_run_name

os.environ["DATA_DIR"] = resource_filename("nerblackbox", f"tests/test_data")


class TestExperiment:

    experiment = Experiment(
        experiment_name="test_experiment",
        from_config=True,
        run_name="runA",
        device="gpu",
        fp16=True,
    )

    # 1 ################################################################################################################
    @pytest.mark.parametrize(
        "runs_name_nr, runs_params, runs_hparams",
        [
            (
                ["runA-1"],
                {
                    "runA-1": {
                        "experiment_name": "test_experiment",
                        "from_config": True,
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
        ],
    )
    def test_create_info_for_runs(
        self,
        runs_name_nr: List[str],
        runs_params: Dict[str, Dict[str, Union[str, int, float, bool]]],
        runs_hparams: Dict[str, Dict[str, Union[str, int, float, bool]]],
    ):
        # 1. runs_name_nr
        assert (
            self.experiment.runs_name_nr == runs_name_nr
        ), f"ERROR! self.experiment.runs_name_nr = {self.experiment.runs_name_nr} != {runs_name_nr}"

        # 2. runs_params
        assert sorted(list(self.experiment.runs_params.keys())) == sorted(
            list(runs_params.keys())
        ), (
            f"ERROR! self.experiment.runs_params.keys() = {sorted(list(self.experiment.runs_params.keys()))} "
            f"does not equal runs_params.keys() = {sorted(list(runs_params.keys()))}"
        )
        keys = list(self.experiment.runs_params.keys())
        for key in keys:
            assert self.experiment.runs_params[key] == runs_params[key], (
                f"ERROR! self.experiment.runs_params[{key}] = {self.experiment.runs_params[key]} "
                f"!= {self.experiment.runs_params[key]}"
            )

    # 2 ################################################################################################################
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
                },
            ),
            (
                "runA",
                {
                    "lr_cooldown_epochs": 1,
                },
            ),
        ],
    )
    def test_create_info_for_mlflow(
        self,
        run_name_nr: Optional[str],
        params_and_hparams: Dict[str, Union[str, int, float, bool]],
    ):
        if run_name_nr is None:
            test_params_and_hparams = self.experiment.params_and_hparams["general"]
        else:
            test_params_and_hparams = self.experiment.params_and_hparams[
                get_run_name(run_name_nr)
            ]
        assert sorted(list(test_params_and_hparams.keys())) == sorted(
            list(params_and_hparams.keys())
        ), (
            f"ERROR! test_params_and_hparams.keys() = {sorted(list(test_params_and_hparams.keys()))} does not equal "
            f"params_and_hparams.keys() = {sorted(list(params_and_hparams.keys()))}"
        )

        keys = list(test_params_and_hparams.keys())
        for key in keys:
            assert test_params_and_hparams[key] == params_and_hparams[key], (
                f"ERROR! test_params_and_hparams[{key}] = {test_params_and_hparams[key]} "
                f"!= {params_and_hparams[key]}"
            )
