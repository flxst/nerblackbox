from typing import Optional, Dict, List, Tuple, Any
from os.path import join, isfile
import pandas as pd
# import mlflow
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.utils.util_functions import epoch2checkpoint
from nerblackbox.modules.ner_training.ner_model_predict import NerModelPredict
from nerblackbox.modules.utils.util_functions import (
    get_run_name,
    compute_mean_and_dmean,
)


class ExperimentResults:

    """class that contains results of a single experiment."""

    def __init__(
        self,
        _id: int,
        name: str,
        experiment: Optional[pd.DataFrame] = None,
        single_runs: Optional[pd.DataFrame] = None,
        average_runs: Optional[pd.DataFrame] = None,
        best_single_run: Optional[Dict] = None,
        best_average_run: Optional[Dict] = None,
        best_model: Optional[NerModelPredict] = None,
    ):
        """

        Args:
            _id: e.g. 1
            name: e.g. 'my_experiment'
            experiment: overview on experiment parameters
            single_runs: overview on run parameters & single  results
            average_runs: overview on run parameters & average results
            best_single_run: overview on best run parameters & single results
            best_average_run: overview on best run parameters & average results
            best_model: best model
        """

        self._id = _id
        self.name = name
        self.experiment = experiment
        self.single_runs = single_runs
        self.average_runs = average_runs
        self.best_single_run = best_single_run
        self.best_average_run = best_average_run
        self.best_model = best_model

    def set_best_model(self, best_model: NerModelPredict) -> None:
        """set best model.

        Args:
            best_model: best model
        """
        self.best_model = best_model

    @classmethod
    def from_mlflow_runs(cls,
                         _runs,
                         _experiment_id,
                         _experiment_name) -> "ExperimentResults":
        experiment_results = ExperimentResults(_id=_experiment_id, name=_experiment_name)
        experiment_results.parse_and_create_dataframe(_runs)  # attr: experiment, single_runs, average_runs
        experiment_results.extract_best_single_run()          # attr: best_single_run
        experiment_results.extract_best_average_run()         # attr: best_average_run
        return experiment_results

    ####################################################################################################################
    # 1. PARSE AND CREATE DATAFRAME -> experiment, single_runs, average_runs
    ####################################################################################################################
    def parse_and_create_dataframe(
        self,
        _runs: List,
    ) -> None:
        r"""
        turn mlflow Run objects (= search_runs() results) into dataframes

        Args:
            _runs:   [list] of [mlflow.entities.Run objects]

        Created Attr:
            experiment:   [pandas DataFrame] overview on experiment parameters
            single_runs:  [pandas DataFrame] overview on single  run parameters & results
            average_runs: [pandas DataFrame] overview on average run parameters & results
        """
        fields_metrics = [
            "epoch_best".upper(),
            "epoch_stopped".upper(),
            "epoch_best_val_entity_fil_f1_micro".upper(),
            "epoch_best_test_entity_fil_f1_micro".upper(),
            "epoch_best_val_token_fil_f1_micro".upper(),
            "epoch_best_test_token_fil_f1_micro".upper(),
            "entity_fil_precision_micro",
            "entity_fil_recall_micro",
        ]

        ###########################################
        # parameters_experiment & parameters_runs
        ###########################################
        parameters_runs, parameters_experiment = self._parse_runs(_runs, fields_metrics)
        self.experiment = pd.DataFrame(parameters_experiment, index=["experiment"]).T

        ###########################################
        # sort
        ###########################################
        by = ("metrics", "epoch_best_val_entity_fil_f1_micro".upper())

        # single runs
        try:
            _single_runs = pd.DataFrame(parameters_runs).sort_values(
                by=by, ascending=False
            )
        except:
            _single_runs = None

        # average runs
        try:
            parameters_runs_average = self.average(parameters_runs)
            _average_runs = pd.DataFrame(parameters_runs_average).sort_values(
                by=by, ascending=False
            )
        except:
            _average_runs = None

        # replace column names
        self.single_runs = self.replace_column_names(_single_runs)
        self.average_runs = self.replace_column_names(_average_runs)

    @classmethod
    def _parse_runs(cls, _runs, _fields_metrics) -> Tuple[Dict[Tuple, Any], Dict[Tuple, Any]]:
        _parameters_runs = dict()
        _parameters_experiment = dict()
        for i in range(len(_runs)):
            if len(_runs[i].data.metrics) == 0:  # experiment
                _parameters_experiment = {
                    k: [v] for k, v in _runs[i].data.params.items()
                }
            else:  # run
                if ("info", "run_id") not in _parameters_runs.keys():
                    _parameters_runs[("info", "run_id")] = [_runs[i].info.run_id]
                else:
                    _parameters_runs[("info", "run_id")].append(_runs[i].info.run_id)

                if ("info", "run_name_nr") not in _parameters_runs.keys():
                    _parameters_runs[("info", "run_name_nr")] = [
                        _runs[i].data.tags["mlflow.runName"]
                    ]
                else:
                    _parameters_runs[("info", "run_name_nr")].append(
                        _runs[i].data.tags["mlflow.runName"]
                    )

                for k, v in _runs[i].data.params.items():
                    if ("params", k) not in _parameters_runs.keys():
                        _parameters_runs[("params", k)] = [v]
                    else:
                        _parameters_runs[("params", k)].append(v)

                for k in _fields_metrics:
                    if ("metrics", k) not in _parameters_runs.keys():
                        try:
                            _parameters_runs[("metrics", k)] = [_runs[i].data.metrics[k]]
                        except:
                            _parameters_runs[("metrics", k)] = [-1]
                    else:
                        try:
                            _parameters_runs[("metrics", k)].append(
                                _runs[i].data.metrics[k]
                            )
                        except:
                            _parameters_runs[("metrics", k)] = [-1]

        for k in ["epoch_best".upper(), "epoch_stopped".upper()]:
            try:
                _parameters_runs[("metrics", k)] = [
                    int(elem) for elem in _parameters_runs[("metrics", k)]
                ]
            except:
                _parameters_runs[("metrics", k)] = [-1]

        return _parameters_runs, _parameters_experiment

    @classmethod
    def average(cls,
                _parameters_runs):
        _parameters_runs_average = {("info", "run_name"): list()}
        _parameters_runs_average.update(
            {
                k: list()
                for k in _parameters_runs.keys()
                if k[0] not in ["info", "metrics"]
            }
        )
        _parameters_runs_average.update(
            {
                k: list()
                for k in [
                    ("metrics", "epoch_best_val_entity_fil_f1_micro".upper()),
                    ("metrics", "d_epoch_best_val_entity_fil_f1_micro".upper()),
                    ("metrics", "epoch_best_test_entity_fil_f1_micro".upper()),
                    ("metrics", "d_epoch_best_test_entity_fil_f1_micro".upper()),
                    ("metrics", "epoch_best_val_token_fil_f1_micro".upper()),
                    ("metrics", "d_epoch_best_val_token_fil_f1_micro".upper()),
                    ("metrics", "epoch_best_test_token_fil_f1_micro".upper()),
                    ("metrics", "d_epoch_best_test_token_fil_f1_micro".upper()),
                    ("metrics", "entity_fil_precision_micro"),
                    ("metrics", "d_entity_fil_precision_micro"),
                    ("metrics", "entity_fil_recall_micro"),
                    ("metrics", "d_entity_fil_recall_micro"),
                ]
            }
        )

        runs_name_nr = _parameters_runs[("info", "run_name_nr")]
        nr_runs = len(runs_name_nr)
        runs_name = list(
            set([get_run_name(run_name_nr) for run_name_nr in runs_name_nr])
        )
        indices = {
            run_name: [
                idx
                for idx in range(nr_runs)
                if get_run_name(runs_name_nr[idx]) == run_name
            ]
            for run_name in runs_name
        }

        #######################
        # loop over runs_name
        #######################
        for run_name in runs_name:
            # add ('info', 'run_name')
            _parameters_runs_average[("info", "run_name")].append(run_name)

            # add ('params', *)
            random_index = indices[run_name][0]
            keys_kept = [k for k in _parameters_runs.keys() if k[0] == "params"]
            for key in keys_kept:
                _parameters_runs_average[key].append(
                    _parameters_runs[key][random_index]
                )

            # add ('metrics', *)
            for level in ["entity", "token"]:
                val_mean, val_dmean = cls.get_mean_and_dmean(indices[run_name], _parameters_runs, "val", level, "f1")
                test_mean, test_dmean = cls.get_mean_and_dmean(indices[run_name], _parameters_runs, "test", level, "f1")
                metrics = {
                    f"epoch_best_val_{level}_fil_f1_micro".upper(): val_mean,
                    f"d_epoch_best_val_{level}_fil_f1_micro".upper(): val_dmean,
                    f"epoch_best_test_{level}_fil_f1_micro".upper(): test_mean,
                    f"d_epoch_best_test_{level}_fil_f1_micro".upper(): test_dmean,
                }
                if level == "entity":
                    test_precision_mean, test_precision_dmean = cls.get_mean_and_dmean(indices[run_name],
                                                                                       _parameters_runs,
                                                                                       "test",
                                                                                       level,
                                                                                       "precision")
                    test_recall_mean, test_recall_dmean = cls.get_mean_and_dmean(indices[run_name],
                                                                                 _parameters_runs,
                                                                                 "test",
                                                                                 level,
                                                                                 "recall")
                    metrics[f"{level}_fil_precision_micro"] = test_precision_mean
                    metrics[f"d_{level}_fil_precision_micro"] = test_precision_dmean
                    metrics[f"{level}_fil_recall_micro"] = test_recall_mean
                    metrics[f"d_{level}_fil_recall_micro"] = test_recall_dmean
                for k in metrics.keys():
                    key = ("metrics", k)
                    _parameters_runs_average[key].append(metrics[k])

        return _parameters_runs_average

    @staticmethod
    def get_mean_and_dmean(_indices_run_name, _parameters_runs, phase, _level, _metric):
        if _metric == "f1":
            metrics_string = f"epoch_best_{phase}_{_level}_fil_{_metric}_micro".upper()
        else:
            metrics_string = f"{_level}_fil_{_metric}_micro"

        values = [
            _parameters_runs[
                ("metrics", metrics_string)
            ][idx]
            for idx in _indices_run_name
        ]
        return compute_mean_and_dmean(values)

    @staticmethod
    def replace_column_names(_runs):
        if _runs is not None:
            _runs.columns = _runs.columns.values
            fields = [elem[1] for elem in _runs.columns if elem[0] == "metrics"]
            for field in fields:
                _runs.rename(
                    columns={
                        ("metrics", field): ("metrics",
                                             field
                                             .replace("EPOCH_BEST_", "")
                                             .replace("_FIL_F1_MICRO", "_F1")
                                             .replace("d_", "D_")
                                             .replace("entity_", "TEST_ENT_")
                                             .replace("ENTITY", "ENT")
                                             .replace("TOKEN", "TOK")
                                             .replace("_fil_precision_micro", "_PRE")
                                             .replace("_fil_recall_micro", "_REC")
                                             ),
                        ("params", "max_seq_length"): ("params", "max_seq"),
                        ("params", "lr_schedule"): ("params", "lr_sch"),
                        ("params", "batch_size"): ("params", "batch_s"),
                    },
                    inplace=True,
                )
            _runs.columns = pd.MultiIndex.from_tuples(_runs.columns)
        return _runs

    ####################################################################################################################
    # 2a. EXTRACT BEST SINGLE RUN -> best_single_run
    ####################################################################################################################
    def extract_best_single_run(self) -> None:
        if self.experiment is not None and self.single_runs is not None:
            _df_best_single_run = self.single_runs.iloc[0, :]
            best_single_run_id = _df_best_single_run[("info", "run_id")]
            best_single_run_name_nr = _df_best_single_run[("info", "run_name_nr")]
            best_single_run_epoch_best = _df_best_single_run[
                ("metrics", "epoch_best".upper())
            ]
            best_single_run_epoch_best_val_entity_fil_f1_micro = _df_best_single_run[
                ("metrics", "val_ent_f1".upper())
            ]
            best_single_run_epoch_best_test_entity_fil_f1_micro = _df_best_single_run[
                ("metrics", "test_ent_f1".upper())
            ]
            best_single_run_epoch_best_test_entity_fil_precision_micro = _df_best_single_run[
                ("metrics", "test_ent_pre".upper())
            ]
            best_single_run_epoch_best_test_entity_fil_recall_micro = _df_best_single_run[
                ("metrics", "test_ent_rec".upper())
            ]

            checkpoint = join(
                env_variable("DIR_CHECKPOINTS"),
                self.name,
                best_single_run_name_nr,
                epoch2checkpoint(best_single_run_epoch_best),
            )

            self.best_single_run = {
                "exp_id": self._id,
                "exp_name": self.name,
                "run_id": best_single_run_id,
                "run_name_nr": best_single_run_name_nr,
                "val_ent_f1".upper(): best_single_run_epoch_best_val_entity_fil_f1_micro,
                "test_ent_f1".upper(): best_single_run_epoch_best_test_entity_fil_f1_micro,
                "test_ent_pre".upper(): best_single_run_epoch_best_test_entity_fil_precision_micro,
                "test_ent_rec".upper(): best_single_run_epoch_best_test_entity_fil_recall_micro,
                "checkpoint": checkpoint if isfile(checkpoint) else None,
            }
        else:
            self.best_single_run = dict()

    ####################################################################################################################
    # 2b. EXTRACT BEST AVERAGE RUN -> best_average_run
    ####################################################################################################################
    def extract_best_average_run(self) -> None:
        if self.experiment is not None and self.average_runs is not None:
            _df_best_average_run = self.average_runs.iloc[0, :]
            best_average_run_name = _df_best_average_run[("info", "run_name")]

            # entity
            best_average_run_epoch_best_val_entity_fil_f1_micro = _df_best_average_run[
                ("metrics", "val_ent_f1".upper())
            ]
            d_best_average_run_epoch_best_val_entity_fil_f1_micro = (
                _df_best_average_run[
                    ("metrics", "d_val_ent_f1".upper())
                ]
            )
            best_average_run_epoch_best_test_entity_fil_f1_micro = _df_best_average_run[
                ("metrics", "test_ent_f1".upper())
            ]
            d_best_average_run_epoch_best_test_entity_fil_f1_micro = (
                _df_best_average_run[
                    ("metrics", "d_test_ent_f1".upper())
                ]
            )

            # token
            best_average_run_epoch_best_val_token_fil_f1_micro = _df_best_average_run[
                ("metrics", "val_tok_f1".upper())
            ]
            d_best_average_run_epoch_best_val_token_fil_f1_micro = (
                _df_best_average_run[
                    ("metrics", "d_val_tok_f1".upper())
                ]
            )
            best_average_run_epoch_best_test_token_fil_f1_micro = _df_best_average_run[
                ("metrics", "test_tok_f1".upper())
            ]
            d_best_average_run_epoch_best_test_token_fil_f1_micro = (
                _df_best_average_run[
                    ("metrics", "d_test_tok_f1".upper())
                ]
            )

            # precision/recall
            best_average_run_epoch_best_test_entity_fil_precision_micro = _df_best_average_run[
                ("metrics", "test_ent_pre".upper())
            ]
            d_best_average_run_epoch_best_test_entity_fil_precision_micro = (
                _df_best_average_run[
                    ("metrics", "d_test_ent_pre".upper())
                ]
            )
            best_average_run_epoch_best_test_entity_fil_recall_micro = _df_best_average_run[
                ("metrics", "test_ent_rec".upper())
            ]
            d_best_average_run_epoch_best_test_entity_fil_recall_micro = (
                _df_best_average_run[
                    ("metrics", "d_test_ent_rec".upper())
                ]
            )

            self.best_average_run = {
                "exp_id": self._id,
                "exp_name": self.name,
                "run_name": best_average_run_name,
                "val_tok_f1".upper(): best_average_run_epoch_best_val_token_fil_f1_micro,
                "d_val_tok_f1".upper(): d_best_average_run_epoch_best_val_token_fil_f1_micro,
                "test_tok_f1".upper(): best_average_run_epoch_best_test_token_fil_f1_micro,
                "d_test_tok_f1".upper(): d_best_average_run_epoch_best_test_token_fil_f1_micro,
                "val_ent_f1".upper(): best_average_run_epoch_best_val_entity_fil_f1_micro,
                "d_val_ent_f1".upper(): d_best_average_run_epoch_best_val_entity_fil_f1_micro,
                "test_ent_f1".upper(): best_average_run_epoch_best_test_entity_fil_f1_micro,
                "d_test_ent_f1".upper(): d_best_average_run_epoch_best_test_entity_fil_f1_micro,
                "test_ent_pre".upper(): best_average_run_epoch_best_test_entity_fil_precision_micro,
                "d_test_ent_pre".upper(): d_best_average_run_epoch_best_test_entity_fil_precision_micro,
                "test_ent_rec".upper(): best_average_run_epoch_best_test_entity_fil_recall_micro,
                "d_test_ent_rec".upper(): d_best_average_run_epoch_best_test_entity_fil_recall_micro,
            }
        else:
            self.best_average_run = dict()
