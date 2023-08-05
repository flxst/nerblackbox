from os.path import isfile, join
import glob
from typing import Optional, Any, Dict, Union, Tuple
from pkg_resources import Requirement
from pkg_resources import resource_filename
import mlflow
from nerblackbox.api.utils import Utils
from nerblackbox.api.store import Store
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.utils.parameters import DATASET, MODEL, SETTINGS, HPARAMS
from nerblackbox.modules.experiment_config.preset import get_preset
from nerblackbox.modules.experiment_results import ExperimentResults


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        from_config: bool = False,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        from_preset: Optional[str] = "adaptive",
        pytest: bool = False,
        verbose: bool = False,
        **kwargs_optional: Any,
    ):
        """

        Args:
            experiment_name: e.g. 'my_experiment'
            from_config: True => read parameters from config file (static definition). False => use Experiment  arguments (dynamic definition)
            model: [equivalent to model_name] e.g. 'bert-base-cased'
            dataset: [equivalent to dataset_name] e.g. 'conll2003'
            from_preset: True => use parameters from preset
            pytest: only for testing, don't specify
            verbose: True => verbose output
            **kwargs_optional: parameters
        """
        self.experiment_name = experiment_name
        self.from_preset = from_preset
        self.verbose = verbose

        self.from_config: bool
        self.kwargs: Dict[str, str]
        self.hparams: Optional[Dict[str, str]]
        self.results: Optional[ExperimentResults]

        if not pytest:
            experiment_exists, experiment_results = Store.get_experiment_results_single(
                experiment_name,
                verbose=self.verbose,
            )
            if experiment_exists:
                self.from_config = True
                self.results = experiment_results
                print(f"> experiment = {experiment_name} found, results loaded.")
            else:
                self.from_config = from_config
                self.kwargs, self.hparams = self._parse_arguments(
                    model, dataset, self.from_preset, **kwargs_optional
                )
                self._checks()  # self.hparams, self.from_preset, self.from_config
                self.results = None
                print(
                    f"> experiment = {experiment_name} not found, create new experiment."
                )

    def show_config(self) -> None:
        r"""
        print experiment config
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

    def run(self):
        r"""run a single experiment.

        Note:

        - from_config == True -> experiment config file is used, no other optional arguments will be used

        - from_config == False -> experiment config file is created dynamically, optional arguments will be used

            - model and dataset are mandatory.

            - All other arguments relate to hyperparameters and are optional.
              They are determined using the following hierarchy:

              1) optional argument

              2) from_preset (adaptive, original, stable),
                 which specifies e.g. the hyperparameters "max_epochs", "early_stopping", "lr_schedule"

              3) default experiment configuration
        """
        _parameters = {
            "experiment_name": self.experiment_name,
            "from_config": int(self.from_config),
            "run_name": self.kwargs["run_name"],
            "device": self.kwargs["device"],
            "fp16": self.kwargs["fp16"],
        }

        if self.from_config is False:
            assert self.hparams is not None, f"ERROR! self.hparams is None."
            self._write_config_file(hparams=self.hparams)

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse("nerblackbox"), "nerblackbox"),
            entry_point="run_experiment",
            experiment_name=self.experiment_name,
            parameters=_parameters,
            env_manager="local",
        )

        experiment_exists, self.results = Store.get_experiment_results_single(
            self.experiment_name,
            verbose=self.verbose,
        )
        assert (
            experiment_exists
        ), f"ERROR! experiment = {self.experiment_name} does not exist."
        print("### single runs ###")
        print(self.results.single_runs.T)
        print()
        print("### average runs ###")
        print(self.results.average_runs.T)

    def get_result(
        self,
        metric: str = "f1",
        level: str = "entity",
        label: str = "micro",
        phase: str = "test",
        average: bool = False,
    ) -> Optional[str]:
        r"""

        Args:
            metric: "f1", "precision", "recall"
            level: "entity" or "token"
            label: "micro", "macro", "PER", ..
            phase: "val" or "test"
            average: if True, return average result of all runs. if False, return result of best run.

        Returns:
            result: e.g. "0.9011 +- 0.0023" (average = True) or "0.9045" (average = False)
        """
        if self.results is None:
            print(f"ATTENTION! no results found")
            return None
        else:
            key = f"{phase.upper()}_{level[:3].upper()}_{metric.upper()}"
            base_quantity = (
                self.results.best_average_run
                if average
                else self.results.best_single_run
            )
            assert isinstance(
                base_quantity, dict
            ), f"ERROR! type(base_quantity) = {type(base_quantity)} should be dict."

            if key in base_quantity:
                if isinstance(
                    base_quantity[key], str
                ):  # average = True,  e.g. "0.9011 +- 0.0023"
                    return base_quantity[key]
                elif isinstance(
                    base_quantity[key], float
                ):  # average = False, e.g. 0.9045..
                    return f"{base_quantity[key]:.4f}"
                else:
                    raise Exception(
                        f"ERROR! found result of unexpected type = {type(base_quantity[key])}"
                    )
            else:
                print(f"ATTENTION! no results found")
                return None

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    def _write_config_file(self, hparams: Dict[str, str]) -> None:
        r"""
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
                hparams is not None
            ), f"ERROR! self.hparams is None - _write_key_value() failed."
            if _key in hparams.keys():
                f.write(f"{_key} = {hparams[_key]}\n")

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

    def _parse_arguments(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        from_preset: Optional[str] = "adaptive",
        **kwargs_optional: Any,
    ) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
        kwargs = Utils.process_kwargs_optional(kwargs_optional)

        if model is not None:
            kwargs["pretrained_model_name"] = model
        if dataset is not None:
            kwargs["dataset_name"] = dataset

        _hparams = Utils.extract_hparams(kwargs)
        kwargs["hparams"] = self._process_hparams(_hparams, from_preset)
        kwargs["run_name"] = kwargs["run_name"] if "run_name" in kwargs else ""
        kwargs["device"] = kwargs["device"] if "device" in kwargs else "gpu"
        kwargs["fp16"] = int(kwargs["fp16"]) if "fp16" in kwargs else 0

        if not self.from_config:
            kwargs["from_preset"] = from_preset
            hparams = kwargs.pop("hparams")

            # get rid of keys in kwargs that are present in hparams
            for key in hparams.keys():
                if key in kwargs.keys():
                    kwargs.pop(key)
        else:
            hparams = None

        return kwargs, hparams

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

    def _checks(self):
        """
        checks that
        - either static or dynamic approach is used, not a mixture
        - if static, that experiment config exists
        - if dynamic, that both model and dataset are specified
        """
        # assert STATIC or DYNAMIC
        assert (self.hparams is None and self.from_config is True) or (
            self.hparams is not None and self.from_config is False
        ), (
            f"ERROR! Need to specify "
            f"EITHER hparams (currently {self.hparams}) "
            f"with or without from_preset (currently {self.from_preset}) "
            f"OR from_config (currently {self.from_config})."
        )

        if self.from_config:
            # STATIC - assert that experiment config exists
            path_experiment_config = join(
                env_variable("DIR_EXPERIMENT_CONFIGS"),
                f"{self.experiment_name}.ini",
            )
            if not isfile(path_experiment_config):
                self._exit_gracefully(
                    f"experiment_config = {path_experiment_config} does not exist."
                )
        else:
            # DYNAMIC - assert that model & dataset are specified
            assert self.hparams is not None, f"ERROR! self.hparams is None."
            for field in ["pretrained_model_name", "dataset_name"]:
                if field not in self.hparams.keys():
                    field_displayed = (
                        "model" if field == "pretrained_model_name" else "dataset"
                    )
                    self._exit_gracefully(
                        f"{field_displayed} is not specified but mandatory if dynamic arguments are used."
                    )

    @staticmethod
    def _exit_gracefully(message: str) -> None:
        print(message)
        print("stopped.")
        exit(0)
