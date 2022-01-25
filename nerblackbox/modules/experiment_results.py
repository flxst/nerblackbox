from typing import Optional, Dict, List, Tuple, Any
from os.path import join, isfile
import pandas as pd
from mlflow.entities import Run
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.utils.util_functions import epoch2checkpoint
from nerblackbox.modules.ner_training.ner_model_predict import NerModelPredict
from nerblackbox.modules.utils.util_functions import (
    get_run_name,
    compute_mean_and_dmean,
)


class ExperimentResults:

    """class that contains results of a single experiment."""

    METRICS = {
        "EPOCH_BEST": "EPOCH_BEST",
        "EPOCH_STOPPED": "EPOCH_STOPPED",
        "EPOCH_BEST_VAL_ENTITY_FIL_F1_MICRO": "VAL_ENT_F1",
        "EPOCH_BEST_TEST_ENTITY_FIL_F1_MICRO": "TEST_ENT_F1",
        "EPOCH_BEST_VAL_TOKEN_FIL_F1_MICRO": "VAL_TOK_F1",
        "EPOCH_BEST_TEST_TOKEN_FIL_F1_MICRO": "TEST_TOK_F1",
        "entity_fil_precision_micro": "TEST_ENT_PRE",
        "entity_fil_recall_micro": "TEST_ENT_REC",
        "entity_fil_asr_abidance": "TEST_ENT_ASR_ABI",
        "entity_fil_asr_precision_micro": "TEST_ENT_ASR_PRE",
        "entity_fil_asr_recall_micro": "TEST_ENT_ASR_REC",
        "entity_fil_asr_f1_micro": "TEST_ENT_ASR_F1",
    }
    METRICS_PLUS = dict(**{"EPOCHS": "EPOCHS"}, **METRICS)
    PARAMS = {
        "max_seq_length": "max_seq",
        "lr_schedule": "lr_sch",
        "batch_size": "batch_sz",
    }

    def __init__(
        self,
        _id: Optional[str] = None,
        name: Optional[str] = None,
        experiment: Optional[pd.DataFrame] = None,
        single_runs: Optional[pd.DataFrame] = None,
        average_runs: Optional[pd.DataFrame] = None,
        best_single_run: Optional[Dict] = None,
        best_average_run: Optional[Dict] = None,
        best_model: Optional[NerModelPredict] = None,
    ):
        """
        Args:
            _id: e.g. '1'
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
    def from_mlflow_runs(
        cls, _runs: List[Run], _experiment_id: str, _experiment_name: str
    ) -> "ExperimentResults":
        """
        Args:
            _runs: [List of mlflow.entities.Run]
            _experiment_id: [str], e.g. '0'
            _experiment_name: [str], e.g. 'my_experiment'

        Returns:
            ExperimentResults instance
        """
        experiment_results = ExperimentResults(
            _id=_experiment_id, name=_experiment_name
        )
        experiment_results._parse_and_create_dataframe(
            _runs
        )  # attr: experiment, single_runs, average_runs
        experiment_results.extract_best_single_run()  # attr: best_single_run
        experiment_results.extract_best_average_run()  # attr: best_average_run
        return experiment_results

    ####################################################################################################################
    # 1. PARSE AND CREATE DATAFRAME -> experiment, single_runs, average_runs
    ####################################################################################################################
    def _parse_and_create_dataframe(
        self,
        _runs: List[Run],
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
        ###########################################
        # parameters_experiment & parameters_runs
        ###########################################
        parameters_runs, parameters_experiment = self._parse_runs(_runs)

        ###########################################
        # rename
        ###########################################
        parameters_runs_renamed = self._rename_parameters_runs(parameters_runs)

        ###########################################
        # average
        ###########################################
        parameters_runs_renamed_average = self._average(parameters_runs_renamed)

        ###########################################
        # dataframes
        ###########################################
        self.experiment = pd.DataFrame(parameters_experiment, index=["experiment"]).T
        self.single_runs, self.average_runs = self._2_sorted_dataframes(
            parameters_runs_renamed, parameters_runs_renamed_average
        )

    @classmethod
    def _parse_runs(
        cls, _runs: List[Run]
    ) -> Tuple[Dict[Tuple[str, str], Any], Dict[str, Any]]:
        """
        Args:
            _runs: list of mlflow.entities.Run objects

        Returns:
            _parameters_runs:       information on single runs
            _parameters_experiment: information on whole experiment
        """
        _parameters_runs: Dict[Tuple[str, str], Any] = dict()
        _parameters_experiment: Dict[str, Any] = dict()
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

                for k in cls.METRICS_PLUS.keys():
                    if ("metrics", k) not in _parameters_runs.keys():
                        try:
                            _parameters_runs[("metrics", k)] = [
                                _runs[i].data.metrics[k]
                            ]
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

        if _parameters_runs[("metrics", "epoch_best".upper())] != [-1]:
            _parameters_runs[("metrics", "epochs".upper())] = [
                elem + 1 for elem in _parameters_runs[("metrics", "epoch_best".upper())]
            ]

        return _parameters_runs, _parameters_experiment

    @classmethod
    def _rename_parameters_runs_single(cls, _tuple: Tuple[str, str]) -> Tuple[str, str]:
        """
        Args:
            _tuple: e.g. ("metrics", "D_entity_fil_precision_micro")

        Returns:
            _tuple_renamed: ("metrics", "TEST_ENT_PRE")
        """
        _category, _field = _tuple
        if _category == "params":
            return (
                "params",
                cls.PARAMS[_field] if _field in cls.PARAMS.keys() else _field,
            )
        elif _category == "metrics":
            if _field in cls.METRICS_PLUS.keys():
                return "metrics", cls.METRICS_PLUS[_field]
            elif _field.replace("D_", "") in cls.METRICS_PLUS.keys():
                return "metrics", f"D_{cls.METRICS_PLUS[_field.replace('D_', '')]}"
            else:
                raise Exception(f"ERROR! _rename_parameters_runs_single() failed.")
        else:
            return _category, _field

    @classmethod
    def _rename_parameters_runs(
        cls, _parameters_runs: Dict[Tuple[str, str], Any]
    ) -> Dict[Tuple[str, str], Any]:
        return {
            cls._rename_parameters_runs_single(k): v
            for k, v in _parameters_runs.items()
        }

    @classmethod
    def _average(
        cls, _parameters_runs_renamed: Dict[Tuple[str, str], Any]
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        Args:
            _parameters_runs_renamed:         information on single runs

        Returns:
            _parameters_runs_renamed_average: information on averaged single runs
        """
        keys = (
            [("info", "run_name")]
            + [
                k
                for k in _parameters_runs_renamed.keys()
                if k[0] not in ["info", "metrics"]
            ]
            + [
                ("metrics", k)
                for k in ["CONVERGENCE"] + list(cls.METRICS_PLUS.values())
            ]
        )
        _parameters_runs_renamed_average: Dict[Tuple[str, str], List[str]] = {
            k: list() for k in keys
        }

        runs_name_nr = _parameters_runs_renamed[("info", "run_name_nr")]
        nr_runs = len(runs_name_nr)
        runs_name = list(
            set([get_run_name(run_name_nr) for run_name_nr in runs_name_nr])
        )
        run_name_2_indices = {
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
            _parameters_runs_renamed_average[("info", "run_name")].append(run_name)

            indices = run_name_2_indices[run_name]

            # add ('params', *)
            random_index = indices[0]
            keys_kept = [k for k in _parameters_runs_renamed.keys() if k[0] == "params"]
            for key in keys_kept:
                _parameters_runs_renamed_average[key].append(
                    _parameters_runs_renamed[key][random_index]
                )

            # add ('metrics', *)
            metrics = dict()

            indices_converged = [
                idx
                for idx in indices
                if _parameters_runs_renamed[
                    ("metrics", cls.METRICS["EPOCH_BEST_VAL_ENTITY_FIL_F1_MICRO"])
                ][idx]
                > 0
            ]
            convergence_str = f"{len(indices_converged)}/{len(indices)}"
            _parameters_runs_renamed_average[("metrics", "CONVERGENCE")].append(
                convergence_str
            )

            for _metric in cls.METRICS_PLUS.values():
                _mean, _dmean = cls._get_mean_and_dmean(
                    indices_converged,
                    _parameters_runs_renamed,
                    _metric=_metric,
                )
                metrics[_metric] = f"{_mean:.5f} +- "
                metrics[_metric] += f"{_dmean:.5f}" if _dmean is not None else "###"
            for k in metrics.keys():
                key = ("metrics", k)
                _parameters_runs_renamed_average[key].append(metrics[k])

        return _parameters_runs_renamed_average

    @classmethod
    def _get_mean_and_dmean(
        cls,
        _indices: List[int],
        _parameters_runs_renamed: Dict[Tuple[str, str], Any],
        _metric: str,
    ) -> Tuple[float, Optional[float]]:
        try:
            values = [
                _parameters_runs_renamed[("metrics", _metric)][idx] for idx in _indices
            ]
        except IndexError:
            values = [-1 for _ in _indices]
        return compute_mean_and_dmean(values)

    def _2_sorted_dataframes(
        self,
        _parameters_runs_renamed: Dict[Tuple[str, str], Any],
        _parameters_runs_renamed_average: Dict[Tuple[str, str], Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        by = ("metrics", self.METRICS["EPOCH_BEST_VAL_ENTITY_FIL_F1_MICRO"])

        # single runs
        try:
            _single_runs = pd.DataFrame(_parameters_runs_renamed).sort_values(
                by=by, ascending=False
            )
        except:
            _single_runs = None

        # average runs
        try:
            _average_runs = pd.DataFrame(_parameters_runs_renamed_average).sort_values(
                by=by, ascending=False
            )
        except:
            _average_runs = None

        return _single_runs, _average_runs

    ####################################################################################################################
    # 2a. EXTRACT BEST SINGLE RUN -> best_single_run
    ####################################################################################################################
    def extract_best_single_run(self) -> None:
        if self.experiment is not None and self.single_runs is not None:
            _df_best_single_run = self.single_runs.iloc[0, :]

            assert (
                self.name is not None
            ), f"ERROR! self.name is None, extract_best_single_run() failed."
            checkpoint = join(
                env_variable("DIR_CHECKPOINTS"),
                self.name,
                _df_best_single_run[("info", "run_name_nr")],
                epoch2checkpoint(_df_best_single_run[("metrics", "EPOCH_BEST")]),
            )

            fields_info = ["run_id", "run_name_nr"]

            self.best_single_run = dict(
                **{
                    "exp_id": self._id,
                    "exp_name": self.name,
                    "checkpoint": checkpoint if isfile(checkpoint) else None,
                },
                **{
                    field: _df_best_single_run[("info", field)] for field in fields_info
                },
                **{
                    field: _df_best_single_run[("metrics", field)]
                    for field in self.METRICS_PLUS.values()
                },
            )
        else:
            self.best_single_run = dict()

    ####################################################################################################################
    # 2b. EXTRACT BEST AVERAGE RUN -> best_average_run
    ####################################################################################################################
    def extract_best_average_run(self) -> None:
        if self.experiment is not None and self.average_runs is not None:
            _df_best_average_run = self.average_runs.iloc[0, :]

            self.best_average_run = dict(
                **{
                    "exp_id": self._id,
                    "exp_name": self.name,
                    "run_name": _df_best_average_run[("info", "run_name")],
                },
                **{
                    field: _df_best_average_run[("metrics", field)]
                    for field in list(self.METRICS_PLUS.values()) + ["CONVERGENCE"]
                },
            )
        else:
            self.best_average_run = dict()
