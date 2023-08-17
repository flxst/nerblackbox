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
from nerblackbox.modules.training_config.preset import get_preset
from nerblackbox.modules.training_results import TrainingResults


class Training:
    def __init__(
        self,
        training_name: str,
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
            training_name: e.g. 'my_training'
            from_config: True => read parameters from config file (static definition). False => use Training arguments (dynamic definition)
            model: [equivalent to model_name] e.g. 'bert-base-cased'
            dataset: [equivalent to dataset_name] e.g. 'conll2003'
            from_preset: True => use parameters from preset
            pytest: only for testing, don't specify
            verbose: True => verbose output
            **kwargs_optional: parameters
        """
        self.training_name = training_name
        self.from_preset = from_preset
        self.verbose = verbose

        self.from_config: bool
        self.kwargs: Dict[str, str]
        self.hparams: Optional[Dict[str, str]]
        self.results: Optional[TrainingResults]

        if not pytest:
            training_exists, training_results = Store.get_training_results_single(
                training_name,
                verbose=self.verbose,
            )
            if training_exists:
                self.from_config = True
                self.results = training_results
                print(f"> training = {training_name} found, results loaded.")
            else:
                self.from_config = from_config
                self.kwargs, self.hparams = self._parse_arguments(
                    model, dataset, self.from_preset, **kwargs_optional
                )
                self._checks()  # self.hparams, self.from_preset, self.from_config
                self.results = None
                print(
                    f"> training = {training_name} not found, create new training."
                )

    def show_config(self) -> None:
        r"""
        print training config
        """
        from nerblackbox.modules.utils.env_variable import env_variable

        if self.training_name != "all":
            path_training_config = join(
                env_variable("DIR_TRAINING_CONFIGS"), f"{self.training_name}.ini"
            )
            if isfile(path_training_config):
                with open(path_training_config, "r") as file:
                    lines = file.read()

                print(f"> training_config = {path_training_config}")
                print()
                print(lines)
            else:
                print(f"> training_config = {path_training_config} does not exist.")
        else:
            training_configs = glob.glob(
                join(env_variable("DIR_TRAINING_CONFIGS"), "*.ini")
            )
            training_configs = [
                elem.split("/")[-1].strip(".ini") for elem in training_configs
            ]
            training_configs = [
                elem for elem in training_configs if elem != "default"
            ]
            for training_config in training_configs:
                print(training_config)

    def run(self):
        r"""run a single training.

        Note:

        - from_config == True -> training config file is used, no other optional arguments will be used

        - from_config == False -> training config file is created dynamically, optional arguments will be used

            - model and dataset are mandatory.

            - All other arguments relate to hyperparameters and are optional.
              They are determined using the following hierarchy:

              1) optional argument

              2) from_preset (adaptive, original, stable),
                 which specifies e.g. the hyperparameters "max_epochs", "early_stopping", "lr_schedule"

              3) default training configuration
        """
        _parameters = {
            "training_name": self.training_name,
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
            entry_point="run_training",
            experiment_name=self.training_name,
            parameters=_parameters,
            env_manager="local",
        )

        training_exists, self.results = Store.get_training_results_single(
            self.training_name,
            verbose=self.verbose,
        )
        assert (
            training_exists
        ), f"ERROR! training = {self.training_name} does not exist."
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
        average: bool = True,
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
        return Store.parse_training_result_single(self.results, metric, level, label, phase, average)

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    def _write_config_file(self, hparams: Dict[str, str]) -> None:
        r"""
        write config file based on self.hparams
        """
        # assert that config file does not exist
        config_path = join(
            env_variable("DIR_TRAINING_CONFIGS"), f"{self.training_name}.ini"
        )
        assert (
            isfile(config_path) is False
        ), f"ERROR! training config file {config_path} already exists!"

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
            hparams:         [dict], e.g. {'multiple_runs': '2'} with hparams to use          [HIERARCHY:  I]
            from_preset:     [str], e.g. 'adaptive' get training params & hparams from preset [HIERARCHY: II]

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
        - if static, that training config exists
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
            # STATIC - assert that training config exists
            path_training_config = join(
                env_variable("DIR_TRAINING_CONFIGS"),
                f"{self.training_name}.ini",
            )
            if not isfile(path_training_config):
                self._exit_gracefully(
                    f"training_config = {path_training_config} does not exist."
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
