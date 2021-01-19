import os
from os.path import join
from configparser import ConfigParser
from nerblackbox.modules.utils.util_functions import get_hardcoded_parameters
from nerblackbox.modules.utils.util_functions import env_variable
from nerblackbox.modules.utils.util_functions import get_run_name, get_run_name_nr
from itertools import product
from typing import Union, Tuple, Any, Dict, Optional, List


class ExperimentConfig:
    """
    class that parses <experiment_name>.ini files
    """

    _, params, hparams, _ = get_hardcoded_parameters(keys=False)

    def __init__(self, experiment_name: str, run_name: str, device, fp16: bool):
        """
        :param experiment_name: [str],         e.g. 'exp1'
        :param run_name:        [str or None], e.g. 'runA'
        :param device:          [torch device]
        :param fp16:            [bool]
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.device = device
        self.fp16 = fp16

        self.config_path = join(
            env_variable("DIR_EXPERIMENT_CONFIGS"), f"{self.experiment_name}.ini"
        )
        if not os.path.isfile(self.config_path):
            raise Exception(f"config file at {self.config_path} does not exist")

        self.config_path_default = join(
            env_variable("DIR_EXPERIMENT_CONFIGS"), "default.ini"
        )
        if not os.path.isfile(self.config_path_default):
            raise Exception(
                f"default config file at {self.config_path_default} does not exist"
            )

    def get_params_and_hparams(
        self, run_name_nr: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        get dictionary of all parameters & their values that belong to
        either generally    to experiment (run_name == None)
        or     specifically to run        (run_name != None)
        --------------------------------------------------------------
        :param run_name_nr:          [str]  e.g. 'runA-1'
        :return: params_and_hparams: [dict] e.g. {'patience': 2, 'mode': 'min', ..}
        """
        _, config_dict_default = self._get_config(default=True)
        config, config_dict = self._get_config(default=False)

        if run_name_nr is None:
            params_and_hparams = config_dict_default["params"]
            params_and_hparams.update(config_dict_default["hparams"])
            params_and_hparams.update(config_dict["params"])
            params_and_hparams.update(config_dict["hparams"])
        else:
            run_name = get_run_name(run_name_nr)
            params_and_hparams = config_dict[run_name]

        return params_and_hparams

    def parse(self) -> Tuple[List[str], Dict[str, Dict], Dict[str, Dict]]:
        """
        parse <experiment_name>.ini files
        if self.run_name is specified, parse only that run. else parse all runs.
        ------------------------------------------------------------------------
        :return: _runs_name_nr [list] of [str], e.g. ['runA-1', 'runA-2', 'runB-1', 'runB-2']
                 _runs_params  [dict] w/ keys = run [str], values = params [dict],
                                      e.g. {'runA-1': {'patience': 2, 'mode': 'min', ..}, ..}
                 _runs_hparams [dict] w/ keys = run [str], values = hparams [dict],
                                      e.g. {'runA-1': {'lr_max': 2e-5, 'max_epochs': 20, ..}, ..}
        """
        _, config_dict_default = self._get_config(default=True)
        config, config_dict = self._get_config(default=False)
        if (
            "params" in config_dict.keys()
            and "multiple_runs" in config_dict["params"].keys()
        ):
            multiple_runs = config_dict["params"]["multiple_runs"]
        elif (
            "params" in config_dict_default.keys()
            and "multiple_runs" in config_dict_default["params"].keys()
        ):
            multiple_runs = config_dict_default["params"]["multiple_runs"]
        else:
            raise Exception(
                f"multiple runs is neither specified in the experiment config nor in the default config."
            )

        # _params_config & _hparams_config
        _runs_params = dict()
        _runs_hparams = dict()

        if self.run_name is None:  # multiple runs
            run_names = [
                run_name for run_name in config.sections() if run_name.startswith("run")
            ]
        else:
            run_names = [
                run_name for run_name in config.sections() if run_name == self.run_name
            ]
            assert len(run_names) == 1

        _runs_name_nr = list()
        for run_name, run_nr in product(run_names, list(range(1, multiple_runs + 1))):
            run_name_nr = get_run_name_nr(run_name, run_nr)

            # _run_params
            _run_params = {
                "experiment_name": self.experiment_name,
                "run_name": run_name,
                "run_name_nr": run_name_nr,
                "device": self.device,
                "fp16": self.fp16,
                "experiment_run_name_nr": f"{self.experiment_name}/{run_name_nr}",
            }
            _run_params.update(config_dict_default["params"])
            _run_params.update(config_dict["params"])

            # _run_hparams
            _run_hparams = dict()
            _run_hparams.update(config_dict_default["hparams"])
            _run_hparams.update(config_dict["hparams"])

            for k, v in config_dict[run_name].items():
                if k in self.params:
                    _run_params[k] = v
                elif k in self.hparams:
                    _run_hparams[k] = v
                else:
                    raise Exception(f"parameter = {k} is unknown.")

            _runs_name_nr.append(run_name_nr)
            _runs_params[run_name_nr] = _run_params
            _runs_hparams[run_name_nr] = _run_hparams

        assert set(_runs_params.keys()) == set(_runs_hparams.keys())

        return _runs_name_nr, _runs_params, _runs_hparams

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    def _get_config(self, default: bool = False) -> Tuple[ConfigParser, Dict]:
        """
        get ConfigParser instance and derive config dictionary from it
        --------------------------------------------------------------
        :param default:        [bool] if True, get default configuration instead of experiment
        :return: _config:      [ConfigParser instance]
        :return: _config_dict: [dict] w/ keys = sections [str], values = [dict] w/ params: values
        """
        _config = ConfigParser()
        _config.read(self.config_path_default if default else self.config_path)
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
                raise Exception(
                    "cannot derive uncased = True/False from pretrained_model_name."
                )

        return _config, _config_dict

    def _convert(
        self, _input_key: str, _input_value: str
    ) -> Union[str, int, float, bool]:
        """
        convert _input string to str/int/float/bool
        -------------------------------------------
        :param _input_key:        [str],                e.g. 'lr_schedule' or 'prune_ratio_train' or 'checkpoints'
        :param _input_value:      [str],                e.g. 'constant'    or '0.01'              or 'False'
        :return: converted_input: [str/int/float/bool], e.g. 'constant'    or  0.01               or  False
        """
        if _input_key in self.params.keys():
            convert_to = self.params[_input_key]
        elif _input_key in self.hparams.keys():
            convert_to = self.hparams[_input_key]
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
        else:
            raise Exception(f"convert_to = {convert_to} not known.")
