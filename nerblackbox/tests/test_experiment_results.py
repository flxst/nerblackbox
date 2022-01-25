import pandas as pd
from typing import List, Dict, Tuple, Any, Union
from mlflow.entities import Run
from nerblackbox.modules.experiment_results import ExperimentResults

import os
from os.path import join, abspath, dirname, isfile
import pickle
from pkg_resources import resource_filename

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR
TRACKING_URI = resource_filename("nerblackbox", f"data/results/mlruns")
FILE_PATH = join(
    resource_filename("nerblackbox", f"tests/test_data"), "test_mlflow_runs.p"
)


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestExperimentResults:

    # load data
    if isfile(FILE_PATH):
        runs: List[Run] = pickle.load(open(FILE_PATH, "rb"))
    else:
        runs = None  # NO TESTING

    experiment_id: str = "0"
    experiment_name: str = "my_experiment_mlflow_runs"
    experiment_results = ExperimentResults(_id=experiment_id, name=experiment_name)

    # results: intermediate
    true_parameters_experiment: Dict[str, Any] = {
        "early_stopping": ["True"],
        "prune_ratio_test": ["0.2"],
        "seed": ["42"],
        "monitor": ["val_loss"],
        "lr_num_cycles": ["4"],
        "prune_ratio_val": ["0.2"],
        "pretrained_model_name": ["af-ai-center/bert-base-swedish-uncased"],
        "max_epochs": ["3"],
        "mode": ["min"],
        "patience": ["5"],
        "multiple_runs": ["2"],
        "checkpoints": ["True"],
        "lr_warmup_epochs": ["2"],
        "lr_cooldown": ["True"],
        "prune_ratio_train": ["0.2"],
        "annotation_scheme": ["plain"],
        "experiment_name": ["my_experiment_mlflow_runs"],
        "device": ["gpu"],
        "min_delta": ["0.3"],
        "dataset_name": ["swedish_ner_corpus"],
        "run_name": ["''"],
        "uncased": ["True"],
        "fp16": ["0"],
        "logging_level": ["info"],
    }
    true_parameters_runs: Dict[Tuple[str, str], Any] = {
        ("info", "run_id"): ["<run_id_2>", "<run_id_1>"],
        ("info", "run_name_nr"): ["runA-2", "runA-1"],
        ("params", "max_seq_length"): ["64", "64"],
        ("params", "lr_max"): ["2e-05", "2e-05"],
        ("params", "lr_schedule"): ["constant", "constant"],
        ("params", "batch_size"): ["16", "16"],
        ("metrics", "EPOCHS"): [3, 3],
        ("metrics", "EPOCH_BEST"): [2, 2],
        ("metrics", "EPOCH_STOPPED"): [2, 2],
        ("metrics", "EPOCH_BEST_VAL_ENTITY_FIL_F1_MICRO"): [
            0.7276507276507277,
            0.7757575757575756,
        ],
        ("metrics", "EPOCH_BEST_TEST_ENTITY_FIL_F1_MICRO"): [
            0.7388535031847134,
            0.691358024691358,
        ],
        ("metrics", "EPOCH_BEST_VAL_TOKEN_FIL_F1_MICRO"): [
            0.7879699248120301,
            0.8240469208211143,
        ],
        ("metrics", "EPOCH_BEST_TEST_TOKEN_FIL_F1_MICRO"): [
            0.7771260997067448,
            0.7463976945244957,
        ],
        ("metrics", "entity_fil_precision_micro"): [
            0.7981651376146789,
            0.721030042918455,
        ],
        ("metrics", "entity_fil_recall_micro"): [
            0.6877470355731226,
            0.6640316205533597,
        ],
        ("metrics", "entity_fil_asr_abidance"): [1.0, 1.0],
        ("metrics", "entity_fil_asr_precision_micro"): [
            0.7981651376146789,
            0.721030042918455,
        ],
        ("metrics", "entity_fil_asr_recall_micro"): [
            0.6877470355731226,
            0.6640316205533597,
        ],
        ("metrics", "entity_fil_asr_f1_micro"): [0.7388535031847134, 0.691358024691358],
    }
    true_parameters_runs_renamed: Dict[Tuple[str, str], Any] = {
        ("info", "run_id"): ["<run_id_2>", "<run_id_1>"],
        ("info", "run_name_nr"): ["runA-2", "runA-1"],
        ("params", "max_seq"): ["64", "64"],
        ("params", "lr_max"): ["2e-05", "2e-05"],
        ("params", "lr_sch"): ["constant", "constant"],
        ("params", "batch_sz"): ["16", "16"],
        ("metrics", "EPOCHS"): [3, 3],
        ("metrics", "EPOCH_BEST"): [2, 2],
        ("metrics", "EPOCH_STOPPED"): [2, 2],
        ("metrics", "VAL_ENT_F1"): [0.7276507276507277, 0.7757575757575756],
        ("metrics", "TEST_ENT_F1"): [0.7388535031847134, 0.691358024691358],
        ("metrics", "VAL_TOK_F1"): [0.7879699248120301, 0.8240469208211143],
        ("metrics", "TEST_TOK_F1"): [0.7771260997067448, 0.7463976945244957],
        ("metrics", "TEST_ENT_PRE"): [0.7981651376146789, 0.721030042918455],
        ("metrics", "TEST_ENT_REC"): [0.6877470355731226, 0.6640316205533597],
        ("metrics", "TEST_ENT_ASR_ABI"): [1.0, 1.0],
        ("metrics", "TEST_ENT_ASR_PRE"): [0.7981651376146789, 0.721030042918455],
        ("metrics", "TEST_ENT_ASR_REC"): [0.6877470355731226, 0.6640316205533597],
        ("metrics", "TEST_ENT_ASR_F1"): [0.7388535031847134, 0.691358024691358],
    }
    true_parameters_runs_renamed_average: Dict[Tuple[str, str], List[str]] = {
        ("info", "run_name"): ["runA"],
        ("params", "max_seq"): ["64"],
        ("params", "lr_max"): ["2e-05"],
        ("params", "lr_sch"): ["constant"],
        ("params", "batch_sz"): ["16"],
        ("metrics", "CONVERGENCE"): ["2/2"],
        ("metrics", "EPOCHS"): ["3.00000 +- 0.00000"],
        ("metrics", "EPOCH_BEST"): ["2.00000 +- 0.00000"],
        ("metrics", "EPOCH_STOPPED"): ["2.00000 +- 0.00000"],
        ("metrics", "VAL_ENT_F1"): ["0.75170 +- 0.01701"],
        ("metrics", "TEST_ENT_F1"): ["0.71511 +- 0.01679"],
        ("metrics", "VAL_TOK_F1"): ["0.80601 +- 0.01276"],
        ("metrics", "TEST_TOK_F1"): ["0.76176 +- 0.01086"],
        ("metrics", "TEST_ENT_PRE"): ["0.75960 +- 0.02727"],
        ("metrics", "TEST_ENT_REC"): ["0.67589 +- 0.00838"],
        ("metrics", "TEST_ENT_ASR_ABI"): ["1.00000 +- 0.00000"],
        ("metrics", "TEST_ENT_ASR_PRE"): ["0.75960 +- 0.02727"],
        ("metrics", "TEST_ENT_ASR_REC"): ["0.67589 +- 0.00838"],
        ("metrics", "TEST_ENT_ASR_F1"): ["0.71511 +- 0.01679"],
    }

    # results: ExperimentResults attributes
    true_experiment: pd.DataFrame = pd.DataFrame(
        true_parameters_experiment, index=["experiment"]
    ).T
    true_single_runs: pd.DataFrame = pd.DataFrame(
        {
            ("info", "run_id"): {1: "<run_id_2>", 0: "<run_id_1>"},
            ("info", "run_name_nr"): {1: "runA-1", 0: "runA-2"},
            ("params", "max_seq"): {1: "64", 0: "64"},
            ("params", "lr_max"): {1: "2e-05", 0: "2e-05"},
            ("params", "lr_sch"): {1: "constant", 0: "constant"},
            ("params", "batch_sz"): {1: "16", 0: "16"},
            ("metrics", "EPOCHS"): {1: 3, 0: 3},
            ("metrics", "EPOCH_BEST"): {1: 2, 0: 2},
            ("metrics", "EPOCH_STOPPED"): {1: 2, 0: 2},
            ("metrics", "VAL_ENT_F1"): {1: 0.7757575757575756, 0: 0.7276507276507277},
            ("metrics", "TEST_ENT_F1"): {1: 0.691358024691358, 0: 0.7388535031847134},
            ("metrics", "VAL_TOK_F1"): {1: 0.8240469208211143, 0: 0.7879699248120301},
            ("metrics", "TEST_TOK_F1"): {1: 0.7463976945244957, 0: 0.7771260997067448},
            ("metrics", "TEST_ENT_PRE"): {1: 0.721030042918455, 0: 0.7981651376146789},
            ("metrics", "TEST_ENT_REC"): {1: 0.6640316205533597, 0: 0.6877470355731226},
            ("metrics", "TEST_ENT_ASR_ABI"): {1: 1.0, 0: 1.0},
            ("metrics", "TEST_ENT_ASR_PRE"): {
                1: 0.721030042918455,
                0: 0.7981651376146789,
            },
            ("metrics", "TEST_ENT_ASR_REC"): {
                1: 0.6640316205533597,
                0: 0.6877470355731226,
            },
            ("metrics", "TEST_ENT_ASR_F1"): {
                1: 0.691358024691358,
                0: 0.7388535031847134,
            },
        }
    )
    true_average_runs: pd.DataFrame = pd.DataFrame(
        {
            ("info", "run_name"): {0: "runA"},
            ("params", "max_seq"): {0: "64"},
            ("params", "lr_max"): {0: "2e-05"},
            ("params", "lr_sch"): {0: "constant"},
            ("params", "batch_sz"): {0: "16"},
            ("metrics", "CONVERGENCE"): {0: "2/2"},
            ("metrics", "EPOCHS"): {0: "3.00000 +- 0.00000"},
            ("metrics", "EPOCH_BEST"): {0: "2.00000 +- 0.00000"},
            ("metrics", "EPOCH_BEST"): {0: "2.00000 +- 0.00000"},
            ("metrics", "EPOCH_STOPPED"): {0: "2.00000 +- 0.00000"},
            ("metrics", "VAL_ENT_F1"): {0: "0.75170 +- 0.01701"},
            ("metrics", "TEST_ENT_F1"): {0: "0.71511 +- 0.01679"},
            ("metrics", "VAL_TOK_F1"): {0: "0.80601 +- 0.01276"},
            ("metrics", "TEST_TOK_F1"): {0: "0.76176 +- 0.01086"},
            ("metrics", "TEST_ENT_PRE"): {0: "0.75960 +- 0.02727"},
            ("metrics", "TEST_ENT_REC"): {0: "0.67589 +- 0.00838"},
            ("metrics", "TEST_ENT_ASR_ABI"): {0: "1.00000 +- 0.00000"},
            ("metrics", "TEST_ENT_ASR_PRE"): {0: "0.75960 +- 0.02727"},
            ("metrics", "TEST_ENT_ASR_REC"): {0: "0.67589 +- 0.00838"},
            ("metrics", "TEST_ENT_ASR_F1"): {0: "0.71511 +- 0.01679"},
        }
    )
    true_best_single_run: Dict[str, Union[str, float]] = {
        "exp_id": "0",
        "exp_name": "my_experiment_mlflow_runs",
        "run_id": "<run_id_2>",
        "run_name_nr": "runA-1",
        "VAL_ENT_F1": 0.7757575757575756,
        "TEST_ENT_F1": 0.691358024691358,
        "VAL_TOK_F1": 0.8240469208211143,
        "TEST_TOK_F1": 0.7463976945244957,
        "TEST_ENT_PRE": 0.721030042918455,
        "TEST_ENT_REC": 0.6640316205533597,
        "TEST_ENT_ASR_ABI": 1.0,
        "TEST_ENT_ASR_PRE": 0.721030042918455,
        "TEST_ENT_ASR_REC": 0.6640316205533597,
        "TEST_ENT_ASR_F1": 0.691358024691358,
        "checkpoint": "<checkpoint>",
    }
    true_best_average_run: Dict[str, str] = {
        "exp_id": "0",
        "exp_name": "my_experiment_mlflow_runs",
        "run_name": "runA",
        "CONVERGENCE": "2/2",
        "VAL_ENT_F1": "0.75170 +- 0.01701",
        "TEST_ENT_F1": "0.71511 +- 0.01679",
        "VAL_TOK_F1": "0.80601 +- 0.01276",
        "TEST_TOK_F1": "0.76176 +- 0.01086",
        "TEST_ENT_PRE": "0.75960 +- 0.02727",
        "TEST_ENT_REC": "0.67589 +- 0.00838",
        "TEST_ENT_ASR_ABI": "1.00000 +- 0.00000",
        "TEST_ENT_ASR_PRE": "0.75960 +- 0.02727",
        "TEST_ENT_ASR_REC": "0.67589 +- 0.00838",
        "TEST_ENT_ASR_F1": "0.71511 +- 0.01679",
    }

    ####################################################################################################################
    # O. FROM_MLFLOW_RUNS
    ####################################################################################################################
    def test_from_mlflow_runs(self):
        if self.runs is not None:
            test_experiment_results = self.experiment_results.from_mlflow_runs(
                self.runs, self.experiment_id, self.experiment_name
            )
            test_experiment_results.single_runs[("info", "run_id")] = [
                "<run_id_2>",
                "<run_id_1>",
            ]
            test_experiment_results.best_single_run["run_id"] = "<run_id_2>"
            test_experiment_results.best_single_run["checkpoint"] = "<checkpoint>"

            pd.testing.assert_frame_equal(
                test_experiment_results.experiment, self.true_experiment
            ), f"ERROR! test_parse_and_create_dataframe / experiment did not pass test"
            pd.testing.assert_frame_equal(
                test_experiment_results.single_runs, self.true_single_runs
            ), f"ERROR! test_parse_and_create_dataframe / single_runs did not pass test"
            pd.testing.assert_frame_equal(
                test_experiment_results.average_runs, self.true_average_runs
            ), f"ERROR! test_parse_and_create_dataframe / average_runs did not pass test"

            for k in self.true_best_single_run.keys():
                assert (
                    test_experiment_results.best_single_run[k]
                    == self.true_best_single_run[k]
                ), f"ERROR! test_best_single_run[{k}] = {test_experiment_results.best_single_run[k]} != {self.true_best_single_run[k]}"
            for k in self.true_best_average_run.keys():
                assert (
                    test_experiment_results.best_average_run[k]
                    == self.true_best_average_run[k]
                ), f"ERROR! test_best_average_run[{k}] = {test_experiment_results.best_average_run[k]} != {self.true_best_average_run[k]}"

    ####################################################################################################################
    # 1. PARSE AND CREATE DATAFRAME -> experiment, single_runs, average_runs
    ####################################################################################################################
    def test_parse_runs(self):
        if self.runs is not None:
            (
                test_parameters_runs,
                test_parameters_experiment,
            ) = self.experiment_results._parse_runs(self.runs)
            test_parameters_runs[("info", "run_id")] = ["<run_id_2>", "<run_id_1>"]

            for k in self.true_parameters_runs.keys():
                assert (
                    test_parameters_runs[k] == self.true_parameters_runs[k]
                ), f"ERROR! test_parameter_runs[{k}] = {test_parameters_runs[k]} != {self.true_parameters_runs[k]}"

            for k in self.true_parameters_experiment.keys():
                assert (
                    test_parameters_experiment[k] == self.true_parameters_experiment[k]
                ), f"ERROR! test_parameter_experiment[{k}] = {test_parameters_experiment[k]} != {self.true_parameters_experiment[k]}"

    def test_rename_parameters_runs(self):
        if self.runs is not None:
            test_parameters_runs_renamed = (
                self.experiment_results._rename_parameters_runs(
                    self.true_parameters_runs
                )
            )

            for k in self.true_parameters_runs_renamed.keys():
                assert (
                    test_parameters_runs_renamed[k]
                    == self.true_parameters_runs_renamed[k]
                ), f"ERROR! test_parameter_runs_renamed[{k}] = {test_parameters_runs_renamed[k]} != {self.true_parameters_runs_renamed[k]}"

    def test_average(self):
        if self.runs is not None:
            test_parameters_runs_renamed_average = self.experiment_results._average(
                self.true_parameters_runs_renamed
            )
            for k in self.true_parameters_runs_renamed_average.keys():
                assert (
                    test_parameters_runs_renamed_average[k]
                    == self.true_parameters_runs_renamed_average[k]
                ), f"ERROR! test_parameter_runs_renamed_average[{k}] = {test_parameters_runs_renamed_average[k]} != {self.true_parameters_runs_renamed_average[k]}"

    def test_parse_and_create_dataframe(self):
        if self.runs is not None:
            self.experiment_results._parse_and_create_dataframe(
                self.runs
            )  # attr: experiment, single_runs, average_runs
            self.experiment_results.single_runs[("info", "run_id")] = [
                "<run_id_2>",
                "<run_id_1>",
            ]
            pd.testing.assert_frame_equal(
                self.experiment_results.experiment, self.true_experiment
            ), f"ERROR! test_parse_and_create_dataframe / experiment did not pass test"
            pd.testing.assert_frame_equal(
                self.experiment_results.single_runs, self.true_single_runs
            ), f"ERROR! test_parse_and_create_dataframe / single_runs did not pass test"
            pd.testing.assert_frame_equal(
                self.experiment_results.average_runs, self.true_average_runs
            ), f"ERROR! test_parse_and_create_dataframe / average_runs did not pass test"

    ####################################################################################################################
    # 2a. EXTRACT BEST SINGLE RUN -> best_single_run
    ####################################################################################################################
    def test_extract_best_single_run(self):
        if self.runs is not None:
            self.experiment_results.extract_best_single_run()  # attr: best_single_run
            self.experiment_results.best_single_run["checkpoint"] = "<checkpoint>"
            for k in self.true_best_single_run.keys():
                assert (
                    self.experiment_results.best_single_run[k]
                    == self.true_best_single_run[k]
                ), f"ERROR! test_best_single_run[{k}] = {self.experiment_results.best_single_run[k]} != {self.true_best_single_run[k]}"

    ####################################################################################################################
    # 2b. EXTRACT BEST AVERAGE RUN -> best_average_run
    ####################################################################################################################
    def test_extract_best_average_run(self):
        if self.runs is not None:
            self.experiment_results.extract_best_average_run()  # attr: best_average_run
            print("test_best_average_run")
            print(self.experiment_results.best_average_run)
            for k in self.true_best_average_run.keys():
                assert (
                    self.experiment_results.best_average_run[k]
                    == self.true_best_average_run[k]
                ), f"ERROR! test_best_average_run[{k}] = {self.experiment_results.best_average_run[k]} != {self.true_best_average_run[k]}"
