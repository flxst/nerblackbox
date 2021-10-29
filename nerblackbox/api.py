# r"""Python API of the nerblackbox package."""

import os
from os.path import abspath
from typing import Optional, Dict, List, Union, Tuple, Any
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

    def run_experiment(self,
                       experiment_name: str,
                       model: Optional[str] = None,
                       dataset: Optional[str] = None,
                       **kwargs_optional: Dict):
        """run a single experiment.

           Note:

           - experiment config file exists -> only experiment_name needs to be provided

           - experiment config file needs to be created -> provide hyperparameters as arguments

             The arguments model and dataset are mandatory in that case.

             All other arguments are optional. If not specified, the default hyperparameters are used.

        Args:
            experiment_name: e.g. 'exp0'
            model: if experiment config file is to be created dynamically, e.g. 'bert-base-uncased'
            dataset: if experiment config file is to be created dynamically, e.g. 'conll-2003'
            kwargs_optional: with optional key-value pairs, e.g. \
            {"multiple_runs": [int], "from_preset": [bool], "run_name": [str], "device": [torch device], "fp16": [bool]}
        """

        kwargs = self._process_kwargs_optional(kwargs_optional)

        kwargs["experiment_name"] = experiment_name
        if model is not None:
            kwargs["pretrained_model_name"] = model
        if dataset is not None:
            kwargs["dataset_name"] = dataset

        kwargs["hparams"], kwargs["from_preset"], kwargs["from_config"] = self._extract_hparams_and_from_preset(kwargs)

        for key in kwargs["hparams"].keys():
            kwargs.pop(key)

        if kwargs["hparams"] == {}:
            kwargs["hparams"] = None

        # TODO START: get rid of this
        print("API: kwargs to NerBlackBoxMain")
        print(kwargs)
        print()
        # TODO END: get rid of this
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
        kwargs["from_config"] = True

        nerbb = NerBlackBoxMain("show_experiment_config", **kwargs)
        nerbb.main()

    ####################################################################################################################
    # HELPER
    ####################################################################################################################
    @staticmethod
    def _process_kwargs_optional(_kwargs_optional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    def _extract_hparams_and_from_preset(_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], bool]:
        """
        Args:
            _kwargs: e.g. {"a": 1, "from_preset": "adaptive"}

        Returns:
            _hparams: e.g. {"a": 1}
            _from_preset: e.g. "adaptive"
            _from_config: e.g. False
        """
        # hparams
        exclude_keys = ["experiment_name", "run_name", "device", "fp16", "from_preset"]
        _hparams = {
            _key: _kwargs[_key]
            for _key in [k for k in _kwargs.keys() if k not in exclude_keys]
        }

        # from_preset
        _from_preset = _kwargs["from_preset"] if "from_preset" in _kwargs.keys() else None

        # from_config
        _from_config = True if (len(_hparams) == 0 and _from_preset is None) else False

        return _hparams, _from_preset, _from_config
