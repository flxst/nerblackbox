import os
from os.path import join, isfile, isdir
import glob
import mlflow
import shutil
from sys import exit
from pkg_resources import Requirement
from pkg_resources import resource_filename, resource_isdir
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.experiment_results import ExperimentResults
from nerblackbox.modules.experiment_config.preset import get_preset
from nerblackbox.modules.utils.parameters import DATASET, MODEL, SETTINGS, HPARAMS
from typing import Optional, Tuple, Union, Dict, List

DATASETS_DOWNLOAD = [
    "conll2003",
    "swedish_ner_corpus",
    "sic",
    "swe_nerc",
]


class NerBlackBoxMain:
    def __init__(
        self,
        flag: str,
        usage: str = "cli",
        dataset_name: Optional[str] = None,  # analyze_data & set_up_dataset
        modify: bool = True,  # set_up_dataset
        val_fraction: float = 0.3,  # set_up_dataset
        verbose: bool = False,
        experiment_name: Optional[str] = None,
        hparams: Optional[Dict[str, Union[str, int, bool]]] = None,  # run_experiment
        from_preset: Optional[str] = None,  # run_experiment
        from_config: bool = False,  # run_experiment
        run_name: Optional[str] = None,  # run_experiment
        device: str = "gpu",  # run_experiment
        fp16: bool = False,  # run_experiment
        text_input: Optional[str] = None,  # predict
        ids: Tuple[str, ...] = (),  # get_experiments, get_experiments_results
        as_df: bool = True,  # get_experiments, get_experiments_results
        results: bool = False,  # clear_data
    ):
        """
        :param flag:            [str], e.g. 'analyze_data', 'set_up_dataset', 'run_experiment', ..
        :param usage:           [str] 'cli' or 'api'
        :param dataset_name:    [str] e.g. 'swedish_ner_corpus'
        :param modify:          [bool] if True: modify tags as specified in method modify_ner_tag_mapping()
        :param val_fraction:    [float] e.g. 0.3
        :param verbose:         [bool]
        :param experiment_name: [str], e.g. 'exp0'
        :param hparams:         [dict], e.g. {'multiple_runs': '2'} with hparams to use            [HIERARCHY:  I]
        :param from_preset:     [str], e.g. 'adaptive' get experiment params & hparams from preset [HIERARCHY: II]
        :param from_config:     [bool] if True, get experiment params & hparams from config file   [ALTERNATIVE]
        :param run_name:        [str or None], e.g. 'runA'
        :param device:          [str]
        :param fp16:            [bool]
        :param text_input:      [str], e.g. 'this is some text that needs to be annotated'
        :param ids:             [tuple of str], experiment_ids to include
        :param as_df:           [bool] if True, return pandas DataFrame, else return dict
        :param results:         [bool] if True, clear not only checkpoints but also mlflow, tensorboard and logs
        """
        self._assert_flag(flag)

        os.environ["MLFLOW_TRACKING_URI"] = env_variable("DIR_MLFLOW")

        self.flag = flag
        self.usage = usage
        self.dataset_name = dataset_name  # analyze_data & set_up_dataset
        self.modify = modify  # set_up_dataset
        self.val_fraction = val_fraction  # set_up_dataset
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.hparams: Optional[
            Dict[str, Union[str, int, bool]]
        ] = self._process_hparams(hparams, from_preset)
        self.from_config: bool = from_config
        self.run_name = run_name  # run_experiment
        self.device = device  # run_experiment
        self.fp16 = fp16  # run_experiment
        self.text_input = text_input  # predict
        self.ids = ids  # get_experiments, get_experiments_results
        self.as_df = as_df  # get_experiments, get_experiments_results
        self.results = results  # clear_data

        if self.flag == "run_experiment":
            assert (self.hparams is None and self.from_config is True) or (
                self.hparams is not None and self.from_config is False
            ), (
                f"ERROR! Need to specify "
                f"EITHER hparams (currently {self.hparams}) "
                f"with or without from_preset (currently {from_preset}) "
                f"OR from_config (currently {self.from_config})."
            )

            if self.from_config:
                path_experiment_config = join(
                    env_variable("DIR_EXPERIMENT_CONFIGS"),
                    f"{self.experiment_name}.ini",
                )
                if not isfile(path_experiment_config):
                    self._exit_gracefully(
                        f"experiment_config = {path_experiment_config} does not exist."
                    )
            else:
                assert (
                    self.hparams is not None
                ), f"ERROR! self.hparams is None but needs to be specified if dynamic arguments are used."
                for field in ["pretrained_model_name", "dataset_name"]:
                    if field not in self.hparams.keys():
                        field_displayed = (
                            "model" if field == "pretrained_model_name" else "dataset"
                        )
                        self._exit_gracefully(
                            f"{field_displayed} is not specified but mandatory if dynamic arguments are used."
                        )

        data_dir = env_variable("DATA_DIR")
        if os.path.isdir(data_dir):
            self._set_client_and_get_experiments()
        else:
            # will be set in init() method
            self.client = None
            self.experiment_id2name = None
            self.experiment_name2id = None

    @staticmethod
    def _exit_gracefully(message: str) -> None:
        print(message)
        print("stopped.")
        exit(0)

    ####################################################################################################################
    # MAIN #############################################################################################################
    ####################################################################################################################
    def main(self):

        ################################################################################################################
        # init
        ################################################################################################################
        if self.flag == "init":
            self._create_data_directory()
            self._set_client_and_get_experiments()

        ################################################################################################################
        # download
        ################################################################################################################
        elif self.flag == "download":
            for _dataset_name in DATASETS_DOWNLOAD:
                self.set_up_dataset(_dataset_name)

        ################################################################################################################
        # analyze_data
        ################################################################################################################
        elif self.flag == "analyze_data":
            self._assert_flag_arg("dataset_name")
            self.analyze_data()

        ################################################################################################################
        # set_up_dataset
        ################################################################################################################
        elif self.flag == "set_up_dataset":
            self._assert_flag_arg("dataset_name")
            self.set_up_dataset(self.dataset_name)

        ################################################################################################################
        # show_experiment_config
        ################################################################################################################
        elif self.flag == "show_experiment_config":
            self._assert_flag_arg("experiment_name")
            self.show_experiment_config()

        ################################################################################################################
        # run_experiment
        ################################################################################################################
        elif self.flag == "run_experiment":
            self._assert_flag_arg("experiment_name")
            self.run_experiment()

        ################################################################################################################
        # get_experiment_results
        ################################################################################################################
        elif self.flag == "get_experiment_results":
            self._assert_usage()
            return self.get_experiment_results()

        ################################################################################################################
        # get_experiments
        ################################################################################################################
        elif self.flag == "get_experiments":
            self._assert_usage()
            return self.get_experiments()

        ################################################################################################################
        # predict
        ################################################################################################################
        elif self.flag == "predict":
            self._assert_flag_arg("experiment_name")
            self._assert_flag_arg("text_input")
            return self.predict()

        ################################################################################################################
        # predict
        ################################################################################################################
        elif self.flag == "predict_proba":
            self._assert_flag_arg("experiment_name")
            self._assert_flag_arg("text_input")
            return self.predict_proba()

        ################################################################################################################
        # clear
        ################################################################################################################
        elif self.flag == "clear_data":
            return self.clear_data()

    ####################################################################################################################
    # FLAGS ############################################################################################################
    ####################################################################################################################
    def analyze_data(self) -> None:
        """
        :used attr: dataset_name [str] e.g. 'swedish_ner_corpus'
        :used attr: verbose      [bool]
        """
        _parameters = {
            "ner_dataset": self.dataset_name,
            "verbose": self.verbose,
        }

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse("nerblackbox"), "nerblackbox"),
            entry_point="analyze_data",
            experiment_name="Default",
            parameters=_parameters,
            use_conda=False,
        )

    def clear_data(self) -> None:
        """
        :used attr: clear_all [bool] if True, clear not only checkpoints but also mlflow, tensorboard and logs
        """
        data_dir = env_variable("DATA_DIR")
        results_dir = join(data_dir, "results")
        assert isdir(results_dir), f"directory {results_dir} does not exist."

        # checkpoints
        objects_to_remove = glob.glob(join(results_dir, "checkpoints", "*"))  # list

        # results (mlflow, tensorboard, ..)
        if self.results:
            results_files = (
                glob.glob(join(results_dir, "mlruns", "*"))
                + glob.glob(join(results_dir, "mlruns", ".*"))
                + glob.glob(join(results_dir, "tensorboard", "*"))
                + glob.glob(join(results_dir, "logs.log"))
                + glob.glob(join(results_dir, "*.npy"))
            )
            objects_to_remove.extend(results_files)

        if len(objects_to_remove) == 0:
            print(f"There is no data to remove in {results_dir}")
        else:
            for elem in objects_to_remove:
                print(elem)
            while 1:
                answer = input("Do you want to remove the above files? (y/n) ")
                if answer == "y":
                    for elem in objects_to_remove:
                        if isfile(elem):
                            os.remove(elem)
                        elif isdir(elem):
                            shutil.rmtree(elem, ignore_errors=False)
                        else:
                            raise ValueError(
                                f"object {elem} is neither a file nor a dir and cannot be removed"
                            )
                    print(f"Files removed")
                    break
                elif answer == "n":
                    print(f"No files removed")
                    break
                else:
                    print("Please enter either y or n")

    def get_experiment_results(self) -> Optional[List[ExperimentResults]]:
        """
        Used Attr:
            experiment_name: e.g. 'all', 'exp0'

        Returns:
            experiment_results_list: returned for API, not CLI
        """
        from nerblackbox.modules.ner_training.ner_model_predict import (
            NerModelPredict,
        )

        assert (
            self.experiment_id2name is not None
        ), f"ERROR! self.experiment_id2name is None."
        if self.experiment_name == "all":
            # all experiments
            assert self.ids is not None, f"ERROR! self.ids is None."
            experiments_id2name = self._filter_experiments_by_ids(self.ids)
        elif self.experiment_name in self.experiment_name2id.keys():
            # single experiment
            experiments_id2name = {
                self.experiment_name2id[self.experiment_name]: self.experiment_name
            }
        else:
            print(f"no experiment with experiment_name = {self.experiment_name} found")
            print(f"experiments that were found:")
            print(self.experiment_name2id)
            return [ExperimentResults()]

        # get ExperimentResults
        experiment_results_list: List[ExperimentResults] = list()
        for _id in sorted(list(experiments_id2name.keys())):
            experiment_results_list.append(self._get_single_experiment_results(_id))

        # return
        if self.usage == "cli":
            if self.experiment_name != "all":
                # single experiment
                print("### single runs ###")
                print(experiment_results_list[0].single_runs.T)
                print()
                print("### average runs ###")
                print(experiment_results_list[0].average_runs.T)
            else:
                # all experiments
                best_single_runs_overview = [
                    experiment_results.best_single_run
                    for experiment_results in experiment_results_list
                ]
                for best_single_runs_elem in best_single_runs_overview:
                    if "checkpoint" in best_single_runs_elem.keys():
                        best_single_runs_elem.pop("checkpoint")

                df_single = (
                    pd.DataFrame(best_single_runs_overview)
                    if self.as_df
                    else best_single_runs_overview
                )
                best_average_runs_overview = [
                    experiment_results.best_average_run
                    for experiment_results in experiment_results_list
                ]
                df_average = (
                    pd.DataFrame(best_average_runs_overview)
                    if self.as_df
                    else best_average_runs_overview
                )
                print("### best single runs ###")
                print(df_single.T)
                print()
                print("### best average runs ###")
                print(df_average.T)
            return None
        else:
            if self.experiment_name != "all":
                # single experiment
                if (
                    "checkpoint" in experiment_results_list[0].best_single_run.keys()
                    and experiment_results_list[0].best_single_run["checkpoint"]
                    is not None
                ):
                    best_model = NerModelPredict.load_from_checkpoint(
                        experiment_results_list[0].best_single_run["checkpoint"]
                    )
                    experiment_results_list[0].set_best_model(best_model)

            return experiment_results_list

    def get_experiments(self) -> Optional[Union[pd.DataFrame, Dict]]:
        r"""
        :used attr: ids    [tuple] of [str], e.g. ('4', '5')
        :used attr: as_df  [bool] if True, return [pandas DataFrame], else return [dict]
        :return: experiments_overview [pandas DataFrame] or [dict]
        """
        assert self.ids is not None, f"ERROR! self.ids is None."
        experiments_id2name_filtered = self._filter_experiments_by_ids(self.ids)

        experiments_overview = sorted(
            [
                {"experiment_id": k, "experiment_name": v}
                for k, v in experiments_id2name_filtered.items()
            ],
            key=lambda elem: elem["experiment_id"],
        )
        df = pd.DataFrame(experiments_overview) if self.as_df else experiments_overview
        if self.usage == "cli":
            print("### experiments ###")
            print(df)
            return None
        else:
            return df

    def predict(self) -> Optional[List[List[Dict[str, str]]]]:
        """
        :used attr: experiment_name [str], e.g. 'exp1'
        :used attr: text_input      [str], e.g. 'this is some text that needs to be annotated'
        :return: predictions [list] of [Namespace] with .internal [list] of (word, tag) tuples
                                                   and  .external [list] of (word, tag) tuples
        """
        nerbb = NerBlackBoxMain(
            "get_experiment_results", experiment_name=self.experiment_name, usage="api"
        )
        experiment_results = nerbb.main()
        predictions = experiment_results[0].best_model.predict(self.text_input)
        if self.usage == "cli":
            print(predictions[0])
            return None
        else:
            return predictions

    def predict_proba(self) -> Optional[List[Dict[str, Union[str, Dict]]]]:
        """
        :used attr: experiment_name [str], e.g. 'exp1'
        :used attr: text_input      [str], e.g. 'this is some text that needs to be annotated'
        :return: predictions [list] of [Namespace] with .internal [list] of (word, tag) tuples
                                                   and  .external [list] of (word, tag) tuples
        """
        nerbb = NerBlackBoxMain(
            "get_experiment_results", experiment_name=self.experiment_name, usage="api"
        )
        experiment_results = nerbb.main()
        predictions = experiment_results.best_model.predict_proba(self.text_input)
        if self.usage == "cli":
            print(predictions[0])
            return None
        else:
            return predictions[0]

    def run_experiment(self) -> None:
        """
        :used attr: experiment_name [str],         e.g. 'exp1'
        :used attr: hparams         [dict], e.g. {'multiple_runs': '2'} with hparams to use           [HIERARCHY:  I]
        :used attr: from_config     [bool] if True, get experiment params & hparams from config file  [ALTERNATIVE]
        :used attr: run_name        [str] or None, e.g. 'runA'
        :used attr: device          [str]
        :used attr: fp16            [bool]
        """
        _parameters = {
            "experiment_name": self.experiment_name,
            "from_config": int(self.from_config),
            "run_name": self.run_name if self.run_name else "",
            "device": self.device,
            "fp16": int(self.fp16),
        }

        if self.from_config is False:
            self._write_config_file()

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse("nerblackbox"), "nerblackbox"),
            entry_point="run_experiment",
            experiment_name=self.experiment_name,
            parameters=_parameters,
            use_conda=False,
        )

        self._get_experiments()  # needs to updated to get results from experiment that was run
        self.get_experiment_results()

    def set_up_dataset(self, _dataset_name: str) -> None:
        """
        :param _dataset_name:    [str] e.g. 'swedish_ner_corpus'
        :used attr: modify       [bool] if True: modify tags as specified in method modify_ner_tag_mapping()
        :used attr: val_fraction [float] e.g. 0.3
        :used attr: verbose      [bool]
        """

        _parameters = {
            "ner_dataset": _dataset_name,
            "modify": self.modify,
            "val_fraction": self.val_fraction,
            "verbose": self.verbose,
        }

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse("nerblackbox"), "nerblackbox"),
            entry_point="set_up_dataset",
            experiment_name="Default",
            parameters=_parameters,
            use_conda=False,
        )

    def show_experiment_config(self) -> None:
        """
        print experiment config
        -----------------------
        :used attr: experiment_name: [str], e.g. 'exp0' or 'all'
        """
        from nerblackbox.modules.utils.env_variable import env_variable

        if self.experiment_name != "all":
            path_experiment_config = join(
                env_variable("DIR_EXPERIMENT_CONFIGS"), f"{self.experiment_name}.ini"
            )
            if isfile(path_experiment_config):
                with open(path_experiment_config, "r") as file:
                    lines = file.read()

                print(f"> experiment_config = {path_experiment_config}")
                print()
                print(lines)
            else:
                print(f"> experiment_config = {path_experiment_config} does not exist.")
        else:
            experiment_configs = glob.glob(
                join(env_variable("DIR_EXPERIMENT_CONFIGS"), "*.ini")
            )
            experiment_configs = [
                elem.split("/")[-1].strip(".ini") for elem in experiment_configs
            ]
            experiment_configs = [
                elem for elem in experiment_configs if elem != "default"
            ]
            for experiment_config in experiment_configs:
                print(experiment_config)

    ####################################################################################################################
    # HELPER ###########################################################################################################
    ####################################################################################################################
    @staticmethod
    def _process_hparams(
        hparams: Optional[Dict[str, Union[str, int, bool]]], from_preset: Optional[str]
    ) -> Optional[Dict[str, Union[str, int, bool]]]:
        """
        Args:
            hparams:         [dict], e.g. {'multiple_runs': '2'} with hparams to use            [HIERARCHY:  I]
            from_preset:     [str], e.g. 'adaptive' get experiment params & hparams from preset [HIERARCHY: II]

        Returns:
            _hparams:        [dict], e.g. {'multiple_runs': '2'} with hparams to use
        """
        _hparams = get_preset(from_preset)
        if _hparams is None:
            _hparams = hparams
        elif hparams is not None:
            _hparams.update(**hparams)
        return _hparams

    def _write_config_file(self) -> None:
        """
        write config file based on self.hparams
        """
        # assert that config file does not exist
        config_path = join(
            env_variable("DIR_EXPERIMENT_CONFIGS"), f"{self.experiment_name}.ini"
        )
        assert (
            isfile(config_path) is False
        ), f"ERROR! experiment config file {config_path} already exists!"

        # write config file: helper functions
        def _write(_str: str):
            f.write(_str + "\n")

        def _write_key_value(_key: str):
            assert (
                self.hparams is not None
            ), f"ERROR! self.hparams is None - _write_key_value() failed."
            if _key in self.hparams.keys():
                f.write(f"{_key} = {self.hparams[_key]}\n")

        # write config file
        with open(config_path, "w") as f:
            _write("[dataset]")
            for key in DATASET.keys():
                _write_key_value(key)

            _write("\n[model]")
            for key in MODEL.keys():
                _write_key_value(key)

            _write("\n[settings]")
            for key in SETTINGS.keys():
                _write_key_value(key)

            _write("\n[hparams]")
            for key in HPARAMS.keys():
                _write_key_value(key)

            _write("\n[runA]")

    @staticmethod
    def _create_data_directory() -> None:
        if resource_isdir(Requirement.parse("nerblackbox"), "nerblackbox/modules/data"):
            data_source = resource_filename(
                Requirement.parse("nerblackbox"), "nerblackbox/modules/data"
            )

            data_dir = env_variable("DATA_DIR")
            print("data_source =", data_source)
            print("data_target =", data_dir)
            if os.path.isdir(data_dir):
                print(f"init: target {data_dir} already exists")
            else:
                shutil.copytree(data_source, data_dir)
                print(f"init: target {data_dir} created")
        else:
            print("init not executed successfully")
            exit(0)

    def _set_client_and_get_experiments(self) -> None:
        """
        :created attr: client             [Mlflow client]
        :created attr: experiment_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        :created attr: experiment_name2id [dict] w/ keys = experiment_name [str] & values = experiment_id [str]
        :return: -
        """
        self.client = MlflowClient()
        self._get_experiments()

    def _get_experiments(self) -> None:
        """
        :created attr: experiment_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        :created attr: experiment_name2id [dict] w/ keys = experiment_name [str] & values = experiment_id [str]
        :return: -
        """
        assert self.client is not None, f"ERROR! self.client is None."
        self.experiment_id2name = {
            elem["_experiment_id"]: elem["_name"]
            for elem in [
                vars(experiment) for experiment in self.client.list_experiments()
            ]
            if elem["_name"] != "Default"
        }

        assert (
            self.experiment_id2name is not None
        ), f"ERROR! self.experiment_id2name is None."
        self.experiment_name2id = {v: k for k, v in self.experiment_id2name.items()}

    ####################################################################################################################
    # HELPER: SINGLE EXPERIMENT
    ####################################################################################################################
    def _get_single_experiment_results(self, experiment_id: str) -> ExperimentResults:
        r"""
        :param experiment_id: [str], e.g. '0'
        :return: experiment_results: [ExperimentResults]
        """
        assert (
            self.experiment_id2name is not None
        ), f"ERROR! self.experiment_id2name is None."

        experiment_name: str = self.experiment_id2name[experiment_id]
        runs: List[Run] = self.client.search_runs(experiment_id)

        return ExperimentResults.from_mlflow_runs(
            runs,
            experiment_id,
            experiment_name,
        )

    ####################################################################################################################
    # HELPER: ALL EXPERIMENTS
    ####################################################################################################################
    def _filter_experiments_by_ids(self, _ids: Tuple[str, ...]) -> Dict[str, str]:
        r"""
        get _experiments_id2name [dict] with _ids as keys
        -------------------------------------------------
        :param _ids:  [tuple] of [str], e.g. ('4', '5')
        :return: _experiments_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        """
        assert (
            self.experiment_id2name is not None
        ), f"ERROR! self.experiment_id2name is None."

        if len(_ids) == 0:
            _experiments_id2name = self.experiment_id2name
        else:
            _experiments_id2name = {
                k: v for k, v in self.experiment_id2name.items() if k in _ids
            }
        return _experiments_id2name

    ####################################################################################################################
    # HELPER: ADDITIONAL
    ####################################################################################################################
    @staticmethod
    def _assert_flag(flag: str) -> None:
        if flag is None:
            message = f"> missing flag (e.g. init, set_up_dataset, run_experiment)"
            print(message)
            exit(0)

    def _assert_flag_arg(self, flag_arg: str) -> None:
        if getattr(self, flag_arg) is None:
            message = f"> missing argument: nerbb {self.flag} <{flag_arg}>"
            print(message)
            exit(0)

    def _assert_usage(self) -> None:
        assert self.usage in ["cli", "api"], "missing usage"

    @staticmethod
    def show_as_df(_dict: Dict) -> pd.DataFrame:
        return pd.DataFrame(_dict)
