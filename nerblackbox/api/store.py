import os
import glob
from typing import List, Union, Dict, Tuple, Optional, Any
from os.path import isdir, abspath, join, isfile
import subprocess
import signal
from pkg_resources import Requirement
from pkg_resources import resource_filename, resource_isdir
import shutil
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.experiment_results import ExperimentResults


class Store:
    r"""
    client for the store that contains all data (datasets, experiment configuration files, models, results)

    Attributes:
         path: path to store's main directory
    """
    path: Optional[str] = os.environ.get("DATA_DIR")

    # will be set by _update_client() & _update_experiments()
    client: Optional[MlflowClient] = None
    experiment_id2name: Optional[Dict[str, str]] = None
    experiment_name2id: Optional[Dict[str, str]] = None

    process: Dict[str, Any] = {
        "mlflow": None,
        "tensorboard": None,
    }

    @classmethod
    def get_path(cls) -> Optional[str]:
        r"""
        Returns:
            cls.path: path to store's main directory
        """
        return cls.path

    @classmethod
    def set_path(cls, path: str) -> Optional[str]:
        r"""
        Args:
            path: path to store's main directory
        Returns:
            cls.path: path to store's main directory
        """
        data_dir = abspath(path)
        os.environ["DATA_DIR"] = data_dir  # environment variable
        cls.path = data_dir  # cls.path
        cls._update_client()  # cls.client
        return cls.get_path()

    @classmethod
    def create(cls, verbose: bool = False) -> None:
        r"""
        create store at cls.path

        Args:
            verbose: output
        """
        if resource_isdir(Requirement.parse("nerblackbox"), "nerblackbox/modules/data"):
            data_source = resource_filename(
                Requirement.parse("nerblackbox"), "nerblackbox/modules/data"
            )

            data_dir = env_variable("DATA_DIR")
            if verbose:
                print("data_source =", data_source)
                print("data_target =", data_dir)

            if isdir(data_dir):
                print(f"> {cls.path} exists.")
            else:
                shutil.copytree(data_source, data_dir)
                print(f"> {cls.path} created.")

            cls._update_client()
        else:
            print("Store.create() not executed successfully")
            exit(0)

    @classmethod
    def show_experiments(
        cls, as_df: bool = True
    ) -> Union[pd.DataFrame, Dict[str, str]]:
        r"""
        Args:
            as_df: if True, return pandas DataFrame. if False, return dict

        Returns:
            experiments: overview of experiments that have been run
        """
        cls._update_experiments()

        assert isinstance(
            cls.experiment_id2name, dict
        ), f"ERROR! type(cls.experiment_id2name) = {type(cls.experiment_id2name)} should be dict."

        experiments_overview = sorted(
            [
                {"experiment_id": k, "experiment_name": v}
                for k, v in cls.experiment_id2name.items()
            ],
            key=lambda elem: elem["experiment_id"],
        )
        return pd.DataFrame(experiments_overview) if as_df else experiments_overview

    @classmethod
    def get_experiment_results(cls) -> List[ExperimentResults]:
        r"""
        get results for all experiments

        Returns:
            experiment_results_list: TODO: instead of list return dict that maps experiment_name to ExperimentResults?
        """
        cls._update_experiments()

        assert (
            cls.experiment_name2id is not None
        ), f"ERROR! cls.experiment_name2id is None."

        # get ExperimentResults
        experiment_results_list: List[ExperimentResults] = list()
        for _name, _id in sorted(
            list(cls.experiment_name2id.items()), key=lambda x: x[1]
        ):  # sort by id
            experiment_exists, experiment_results = cls.get_experiment_results_single(
                _name, update_experiments=False
            )
            assert experiment_exists, f"ERROR! experiment = {_name} does not exist."
            experiment_results_list.append(experiment_results)

        # return
        # CLI skipped
        return experiment_results_list

    @classmethod
    def get_experiment_results_single(
        cls, experiment_name: str, update_experiments: bool = True, verbose: bool = False
    ) -> Tuple[bool, ExperimentResults]:
        r"""
        get results for single experiment

        Args:
            experiment_name: e.g. 'exp0'
            update_experiments: whether to update cls.experiment_id2name & cls.experiment_name2id
            verbose: output

        Returns:
            experiment_results: for experiment with experiment_name
        """
        if cls.client is None:
            cls._update_client()

        if isdir(env_variable("DIR_MLFLOW")):
            if update_experiments:
                cls._update_experiments()

            assert (
                cls.experiment_name2id is not None
            ), f"ERROR! cls.experiment_name2id is None."

            if experiment_name in cls.experiment_name2id.keys():
                experiment_id: str = cls.experiment_name2id[experiment_name]
                assert isinstance(
                    cls.client, MlflowClient
                ), f"ERROR! type(cls.client) = {type(cls.client)} should be MlflowClient"
                runs: List[Run] = cls.client.search_runs([experiment_id])

                return True, ExperimentResults.from_mlflow_runs(
                    runs,
                    experiment_id,
                    experiment_name,
                )
            else:
                if verbose:
                    print(f"no experiment with experiment_name = {experiment_name} found")
                    print(f"experiments that were found:")
                    print(list(cls.experiment_name2id.keys()))
                return False, ExperimentResults()
        else:
            return False, ExperimentResults()

    @classmethod
    def mlflow(cls, action: str) -> None:
        """
        start or stop the mlflow server at http://127.0.0.1:5000 or check its status

        Args:
            action: "start", "status", "stop"
        """
        cls._subprocess(action, "mlflow")

    @classmethod
    def tensorboard(cls, action: str) -> None:
        """
        start or stop the tensorboard server at http://127.0.0.1:6006 or check its status

        Args:
            action: "start", "status", "stop"
        """
        cls._subprocess(action, "tensorboard")

    @classmethod
    def clear_data(cls, results: bool = False) -> None:
        """
        :used attr: clear_all [bool] if True, clear not only checkpoints but also mlflow, tensorboard and logs
        """
        assert isinstance(
            cls.path, str
        ), f"ERROR! type(cls.path) = {type(cls.path)} should be str."
        results_dir = join(cls.path, "results")
        assert isdir(results_dir), f"directory {results_dir} does not exist."

        # checkpoints
        objects_to_remove = glob.glob(join(results_dir, "checkpoints", "*"))  # list

        # results (mlflow, tensorboard, ..)
        if results:
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

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    @classmethod
    def _update_client(cls) -> None:
        """
        Changed Attr:
            client: [MlflowClient]
        """
        os.environ["MLFLOW_TRACKING_URI"] = env_variable("DIR_MLFLOW")

        data_dir_exists_before = isdir(env_variable("DATA_DIR"))
        mlflow_subdirectory_exists_before = isdir(env_variable("DIR_MLFLOW"))
        cls.client = (
            MlflowClient()
        )  # creates initial subdirectory DATA_DIR/results/mlruns/0 (if cli is used)
        mlflow_subdirectory_exists_after = isdir(env_variable("DIR_MLFLOW"))

        if (
            mlflow_subdirectory_exists_before is False
            and mlflow_subdirectory_exists_after is True
        ):
            if data_dir_exists_before is False:
                # if whole DATA_DIR is new => delete whole DATA_DIR
                shutil.rmtree(env_variable("DATA_DIR"))
            else:
                # if only DIR_MLFLOW is new => delete only DIR_MLFLOW
                shutil.rmtree(env_variable("DIR_MLFLOW"))

        if data_dir_exists_before:
            # create DIR_MLFLOW and subdirectory .trash
            os.makedirs(join(env_variable("DIR_MLFLOW"), ".trash"), exist_ok=True)

    @classmethod
    def _update_experiments(cls) -> None:
        """
        Changed Attr:
            experiment_id2name: [dict] w/ keys = experiment_id   [str] & values = experiment_name [str]
            experiment_name2id: [dict] w/ keys = experiment_name [str] & values = experiment_id   [str]
        """
        assert cls.client is not None, f"ERROR! cls.client is None."
        cls.experiment_id2name = {
            elem["_experiment_id"]: elem["_name"]
            for elem in [
                vars(experiment) for experiment in cls.client.search_experiments()
            ]
            if elem["_name"] != "Default"
        }

        assert (
            cls.experiment_id2name is not None
        ), f"ERROR! cls.experiment_id2name is None."
        cls.experiment_name2id = {v: k for k, v in cls.experiment_id2name.items()}

    @classmethod
    def _subprocess(cls, _action: str, server_type: str) -> None:
        """
        start or stop a mlflow/tensorboard subprocess, or check its status
        see https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true

        Args:
            _action: "start", "status", "stop"
            server_type: "mlflow", "tensorboard"
        """
        link = cls._get_link(server_type)
        if _action == "start":
            if cls.process[server_type] is None:
                start_command = cls._get_start_command(server_type)
                cls.process[server_type] = subprocess.Popen(
                    start_command,
                    stdout=subprocess.PIPE,
                    shell=True,
                    preexec_fn=os.setsid,
                )
                print(
                    f"{server_type} process with pid = {cls.process[server_type].pid} started"
                )
                print(link)
            else:
                print(
                    f"{server_type} process with pid = {cls.process[server_type].pid} already running"
                )
                print(link)
        elif _action == "status":
            if cls.process[server_type] is None:
                print(f"no {server_type} process found")
            else:
                print(
                    f"{server_type} process with pid = {cls.process[server_type].pid} running"
                )
                print(link)
        elif _action == "stop":
            if cls.process[server_type] is None:
                print(f"no {server_type} process found")
            else:
                os.killpg(os.getpgid(cls.process[server_type].pid), signal.SIGTERM)
                print(
                    f"{server_type} process with pid = {cls.process[server_type].pid} killed"
                )
                print(link)
                cls.process[server_type] = None

    @classmethod
    def _get_start_command(cls, server_type: str) -> str:
        """
        Args:
            server_type: "mlflow", "tensorboard"

        Returns:
            start_command: e.g. "cd [..]; mlflow ui"
        """
        cmd_cd = f'cd {join(env_variable("DATA_DIR"), "results")}'
        if server_type == "mlflow":
            return f"{cmd_cd}; mlflow ui"
        elif server_type == "tensorboard":
            return f"{cmd_cd}; tensorboard --logdir tensorboard --reload_multifile=true"
        else:
            raise Exception(f"ERROR! server_type = {server_type} unknown.")

    @classmethod
    def _get_link(cls, server_type: str) -> str:
        """
        Args:
            server_type: "mlflow", "tensorboard"

        Returns:
            link: e.g. "see http://127.0.0.1:5000"
        """
        if server_type == "mlflow":
            return f"see http://127.0.0.1:5000"
        elif server_type == "tensorboard":
            return f"see http://127.0.0.1:6006"
        else:
            raise Exception(f"ERROR! server_type = {server_type} unknown.")
