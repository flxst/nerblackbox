import os
from os.path import join
from configparser import ConfigParser
from nerblackbox.modules.utils.parameters import PARAMS, HPARAMS
from nerblackbox.modules.utils.env_variable import env_variable
from typing import Union, Tuple, Any, Dict, List


class ExperimentConfig:
    """
    class that parses <experiment_name>.ini files
    """

    def __init__(self, experiment_name: str):
        """
        Args:
            experiment_name: e.g. 'exp1', 'default
        """
        self.config: Dict[str, Dict[str, str]]
        self.run_names: List[str]
        self.config, self.run_names = self._get_config(experiment_name)

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    def _get_config(
        self, experiment_name: str
    ) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
        """
        get ConfigParser instance and derive config dictionary from it

        Args:
            experiment_name: e.g. 'exp1', 'default

        Returns:
            _config_dict: w/ keys = sections [str], values = [dict] w/ key: value = params: values
            _run_names: e.g. ["runA", "runB"]
        """
        config_path = join(
            env_variable("DIR_EXPERIMENT_CONFIGS"), f"{experiment_name}.ini"
        )
        if not os.path.isfile(config_path):
            raise Exception(f"config file at {config_path} does not exist")

        _config = ConfigParser()
        _config.read(config_path)
        _config_dict: Dict[str, Dict[str, Any]] = {
            s: dict(_config.items(s)) for s in _config.sections()
        }  # {'hparams': {'monitor': 'val_loss'}}
        _config_dict = {
            s: {k: self._convert(k, v) for k, v in subdict.items()}
            for s, subdict in _config_dict.items()
        }

        # combine sections 'dataset', 'model' & 'settings' to single section 'params'
        _config_dict["params"] = dict()
        for s in ["dataset", "model", "settings"]:
            if s in _config_dict.keys():
                _config_dict["params"].update(_config_dict[s])
                _config_dict.pop(s)

        # derive uncased
        if (
            "uncased" not in _config_dict["params"].keys()
            and "pretrained_model_name" in _config_dict["params"]
        ):
            if "uncased" in _config_dict["params"]["pretrained_model_name"]:
                _config_dict["params"]["uncased"] = True
            elif "cased" in _config_dict["params"]["pretrained_model_name"]:
                _config_dict["params"]["uncased"] = False
            else:
                _config_dict["params"]["uncased"] = False
                print(
                    "ATTENTION! could not derive uncased = True/False from pretrained_model_name."
                    " => assume model is cased"
                )

        _run_names = [
            run_name for run_name in _config.sections() if run_name.startswith("run")
        ]

        return _config_dict, _run_names

    @staticmethod
    def _convert(_input_key: str, _input_value: str) -> Union[str, int, float, bool]:
        """
        convert _input string to str/int/float/bool

        Args:
            _input_key:        [str],              e.g. 'lr_schedule' or 'prune_ratio_train' or 'checkpoints'
            _input_value:      [str],              e.g. 'constant'    or '0.01'              or 'False'

        Returns:
            converted_input: [str/int/float/bool], e.g. 'constant'    or  0.01               or  False
        """
        if _input_key in PARAMS.keys():
            convert_to = PARAMS[_input_key]
        elif _input_key in HPARAMS.keys():
            convert_to = HPARAMS[_input_key]
        else:
            raise Exception(f"_input_key = {_input_key} unknown.")

        if convert_to == "str":
            return _input_value
        elif convert_to == "int":
            return int(_input_value)
        elif convert_to == "float":
            return float(_input_value)
        elif convert_to == "bool":
            return _input_value not in ["False", "false"]
        else:  # pragma: no cover
            raise Exception(f"convert_to = {convert_to} not known.")
