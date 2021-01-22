# r"""Python API of the nerblackbox package."""

import os
from os.path import abspath
from typing import Optional, Dict, List, Union
import pandas as pd

try:
    from nerblackbox.modules.main import NerBlackBoxMain
except ModuleNotFoundError as e:
    print(os.getcwd())
    print("- 1 -")
    print(e)
finally:
    try:
        import nerblackbox
    except ModuleNotFoundError as e2:
        print("- 2 -")
        print(e2)


class NerBlackBox:
    """class that provides all nerblackbox functionalities."""

    def __init__(self, base_dir: str = ".", data_dir: str = "./data"):
        """

        Args:
            base_dir: relative path of base directory with respect to current directory
            data_dir: relative path of data directory with respect to current directory

        """

        os.environ["BASE_DIR"] = abspath(base_dir)
        os.environ["DATA_DIR"] = abspath(data_dir)
        print("BASE_DIR = ", os.environ.get("BASE_DIR"))
        print("DATA_DIR = ", os.environ.get("DATA_DIR"))

    ####################################################################################################################
    # NER BLACK BOX
    ####################################################################################################################
    def analyze_data(self, dataset_name: str, **kwargs_optional: Dict):
        """analyze a dataset.

        Args:
            dataset_name: e.g. "swedish_ner_corpus".
            kwargs_optional: with optional key-value pairs {"verbose": [bool]}.
        """

        kwargs = self._process_kwargs_optional(kwargs_optional)
        kwargs["dataset_name"] = dataset_name

        nerbb = NerBlackBoxMain("analyze_data", **kwargs)
        nerbb.main()

    def download(self):
        """download & prepare built-in datasets, prepare experiment configuration.
        needs to be called exactly once before any other CLI/API commands of the package are executed
        in case built-in datasets shall be used.
        """
        _ = self._process_kwargs_optional()

        nerbb = NerBlackBoxMain("download")
        nerbb.main()

    def get_experiment_results(self, experiment_name: str):
        """get results for a single experiment.

        Args:
            experiment_name: e.g. "exp0"
        """
        kwargs = self._process_kwargs_optional()
        kwargs["usage"] = "api"
        kwargs["experiment_name"] = experiment_name

        nerbb = NerBlackBoxMain("get_experiment_results", **kwargs)
        return nerbb.main()

    def get_experiments(self, **kwargs_optional: Dict) -> pd.DataFrame:
        """show list of experiments that have been run.

        Args:
            kwargs_optional: with optional key-value pairs \
            {"ids": [tuple of int], "as_df": [bool]}

        Returns:
            experiments_overview: overview
        """
        kwargs = self._process_kwargs_optional(kwargs_optional)
        kwargs["usage"] = "api"

        nerbb = NerBlackBoxMain("get_experiments", **kwargs)
        return nerbb.main()

    def get_experiments_results(self, **kwargs_optional: Dict) -> Dict:
        """get results from multiple experiments.

        Args:
            kwargs_optional: with optional key-value pairs \
            {"ids": [tuple of int], "as_df": [bool]}

        Returns:
            experiments_results: w/ keys = "best_single_runs", "best_average_runs" \
            & values = [pandas DataFrame] or [dict]
        """
        kwargs = self._process_kwargs_optional(kwargs_optional)
        kwargs["usage"] = "api"
        nerbb = NerBlackBoxMain("get_experiments_results", **kwargs)
        return nerbb.main()

    def init(self):
        """initialize the data_dir directory.
        needs to be called exactly once before any other CLI/API commands of the package are executed.
        """
        _ = self._process_kwargs_optional()

        nerbb = NerBlackBoxMain("init")
        nerbb.main()

    def predict(self, experiment_name: str, text_input: Union[str, List[str]]):
        """predict labels for text_input using the best model of a single experiment.

        Args:
            experiment_name: e.g. "exp0"
            text_input: e.g. "this text needs to be tagged"
        """

        kwargs = self._process_kwargs_optional()
        kwargs["usage"] = "api"
        kwargs["experiment_name"] = experiment_name
        kwargs["text_input"] = text_input

        nerbb = NerBlackBoxMain("predict", **kwargs)
        return nerbb.main()

    def run_experiment(self, experiment_name: str, **kwargs_optional: Dict):
        """run a single experiment.

        Args:
            experiment_name: e.g. "exp0"
            kwargs_optional: with optional key-value pairs \
            {"run_name": [str], "device": [torch device], "fp16": [bool]}
        """

        kwargs = self._process_kwargs_optional(kwargs_optional)
        kwargs["experiment_name"] = experiment_name

        nerbb = NerBlackBoxMain("run_experiment", **kwargs)
        nerbb.main()

    def set_up_dataset(self, dataset_name: str, **kwargs_optional: Dict):
        """set up a dataset using the associated Formatter class.

        Args:
            dataset_name: e.g. "swedish_ner_corpus"
            kwargs_optional: with optional key-value pairs \
            {"modify": [bool], "val_fraction": [float], "verbose": [bool]}
        """

        kwargs = self._process_kwargs_optional(kwargs_optional)
        kwargs["dataset_name"] = dataset_name

        nerbb = NerBlackBoxMain("set_up_dataset", **kwargs)
        nerbb.main()

    def show_experiment_config(self, experiment_name: str):
        """show a single experiment configuration in detail.

        Args:
            experiment_name: e.g. "exp0"
        """
        kwargs = self._process_kwargs_optional()
        kwargs["experiment_name"] = experiment_name

        nerbb = NerBlackBoxMain("show_experiment_config", **kwargs)
        nerbb.main()

    def show_experiment_configs(self):
        """show overview on all available experiment configurations."""
        _ = self._process_kwargs_optional()

        nerbb = NerBlackBoxMain("show_experiment_configs")
        nerbb.main()

    ####################################################################################################################
    # HELPER
    ####################################################################################################################
    @staticmethod
    def _process_kwargs_optional(_kwargs_optional: Optional[Dict] = None):
        if _kwargs_optional is None:
            return {}
        else:
            return {k: v for k, v in _kwargs_optional.items() if v is not None}
