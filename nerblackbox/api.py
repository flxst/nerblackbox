# r"""Python API of the nerblackbox package."""

import os
from os.path import abspath
from typing import Optional, Dict, List, Union, Any
import pandas as pd
from nerblackbox.modules.experiment_results import ExperimentResults

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

    def get_experiment_results(self, experiment_name: str) -> List[ExperimentResults]:
        """get results for a single experiment.

        Args:
            experiment_name: e.g. "exp0"

        Returns:
            see ExperimentResults
        """
        kwargs = self._process_kwargs_optional()
        kwargs["usage"] = "api"
        kwargs["experiment_name"] = experiment_name
        kwargs["from_config"] = True

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

    def run_experiment(
        self,
        experiment_name: str,
        from_config: bool = False,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        from_preset: Optional[str] = "adaptive",
        **kwargs_optional: Dict
    ):
        """run a single experiment.

           Note:

           - from_config == True -> experiment config file is used, no other optional arguments will be used

           - from_config == False -> experiment config file is created dynamically, optional arguments will be used

               - model and dataset are mandatory.

               - All other arguments relate to hyperparameters and are optional.
                 If not specified, they are taken using the following hierarchy:

                 1) optional argument

                 2) from_preset (adaptive, original, stable),
                    which specifies e.g. the hyperparameters "max_epochs", "early_stopping", "lr_schedule"

                 3) default experiment configuration


        Args:
            experiment_name: e.g. 'exp0'
            from_config: e.g. False
            model: if experiment config file is to be created dynamically, e.g. 'bert-base-uncased'
            dataset: if experiment config file is to be created dynamically, e.g. 'conll-2003'
            from_preset: if experiment config file is to be created dynamically, e.g. 'adaptive'
            kwargs_optional: with optional key-value pairs, e.g. \
            {"multiple_runs": [int], "run_name": [str], "device": [torch device], "fp16": [bool]}
        """

        kwargs = self._process_kwargs_optional(kwargs_optional)

        kwargs["experiment_name"] = experiment_name
        if model is not None:
            kwargs["pretrained_model_name"] = model
        if dataset is not None:
            kwargs["dataset_name"] = dataset

        kwargs["hparams"] = self._extract_hparams(kwargs)
        kwargs["from_config"] = from_config
        if not from_config:
            kwargs["from_preset"] = from_preset

        # get rid of keys in kwargs that are present in kwargs["hparams"]
        for key in kwargs["hparams"].keys():
            kwargs.pop(key)

        if kwargs["hparams"] == {}:
            kwargs["hparams"] = None

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
        """show a single experiment configuration in detail
           or an overview on all available experiment configurations.

        Args:
            experiment_name: e.g. "exp0" or "all"
        """
        kwargs = self._process_kwargs_optional()
        kwargs["experiment_name"] = experiment_name

        nerbb = NerBlackBoxMain("show_experiment_config", **kwargs)
        nerbb.main()

    ####################################################################################################################
    # HELPER
    ####################################################################################################################
    @staticmethod
    def _process_kwargs_optional(
        _kwargs_optional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        general helper function
        filters out key-value pairs that have value = None

        Args:
            _kwargs_optional: e.g. {"a": 1, "b": None}

        Returns:
            _kwargs:          e.g. {"a": 1}
        """
        if _kwargs_optional is None:
            return {}
        else:
            return {k: v for k, v in _kwargs_optional.items() if v is not None}

    @staticmethod
    def _extract_hparams(_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            _kwargs: e.g. {"a": 1, "run_name": "runA-1"}

        Returns:
            _hparams: e.g. {"a": 1}
        """
        # hparams
        exclude_keys = ["experiment_name", "run_name", "device", "fp16"]
        _hparams = {
            _key: _kwargs[_key] for _key in _kwargs.keys() if _key not in exclude_keys
        }

        return _hparams
