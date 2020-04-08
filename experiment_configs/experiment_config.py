
import os
from os.path import abspath, dirname
from configparser import ConfigParser
from ner.utils.util_functions import get_hardcoded_parameters


class ExperimentConfig:
    """
    class that parses <experiment_name>.ini files
    """

    _, params, hparams, _ = get_hardcoded_parameters(keys=False)

    def __init__(self,
                 experiment_name: str,
                 run_name: str,
                 device,
                 fp16: bool):
        """
        :param experiment_name: [str],         e.g. exp1
        :param run_name:        [str] or None, e.g. run1
        :param device:          [torch device]
        :param fp16:            [bool]
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.device = device
        self.fp16 = fp16

        self.config_path = f'{abspath(dirname(__file__))}/{self.experiment_name}.ini'
        if not os.path.isfile(self.config_path):
            raise Exception(f'config file at {self.config_path} does not exist')

    def get_params_and_hparams(self, run_name: str = None):
        """
        get dictionary of all parameters & their values that belong to
        either generally    to experiment (run_name == None)
        or     specifically to run        (run_name != None)
        --------------------------------------------------------------
        :param run_name:             [str]  e.g. 'run1'
        :return: params_and_hparams: [dict] e.g. {'patience': 2, 'mode': 'min', ..}
        """
        config, config_dict = self._get_config()

        if run_name is None:
            params_and_hparams = config_dict['params']
            params_and_hparams.update(config_dict['hparams'])
        else:
            params_and_hparams = config_dict[run_name]

        return params_and_hparams

    def parse(self):
        """
        parse <experiment_name>.ini files
        if self.run_name is specified, parse only that run. else parse all runs.
        ------------------------------------------------------------------------
        :return: runs             [list] of [str], e.g. ['run1', 'run2']
                 _run_params  [dict] w/ keys = run_name [str], values = params [dict],
                                     e.g. {'run1': {'patience': 2, 'mode': 'min', ..}, ..}
                 _run_hparams [dict] w/ keys = run_name [str], values = hparams [dict],
                                     e.g. {'run1': {'lr_max': 2e-5, 'max_epochs': 20, ..}, ..}
        """
        config, config_dict = self._get_config()

        # _params_config & _hparams_config
        _run_params = dict()
        _run_hparams = dict()

        if self.run_name is None:  # multiple runs
            runs = [run for run in config.sections() if run.startswith('run')]
        else:
            runs = [run for run in config.sections() if run == self.run_name]
            assert len(runs) == 1

        for run in runs:
            # _run_params
            _run_params[run] = {
                'experiment_name': self.experiment_name,
                'run_name': run,
                'device': self.device,
                'fp16': self.fp16,
                'experiment_run_name': f'{self.experiment_name}/{run}',
            }
            _run_params[run].update(config_dict['params'])

            # _run_hparams
            _run_hparams[run] = config_dict['hparams']

            for k, v in config_dict[run].items():
                if k in self.params:
                    _run_params[run][k] = v
                elif k in self.hparams:
                    _run_hparams[run][k] = v
                else:
                    raise Exception(f'parameter = {k} is unknown.')

        assert set(_run_params.keys()) == set(_run_hparams.keys())

        return runs, _run_params, _run_hparams

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    def _get_config(self):
        """
        get ConfigParser instance and derive config dictionary from it
        --------------------------------------------------------------
        :return: _config:      [ConfigParser instance]
        :return: _config_dict: [dict] w/ keys = sections [str], values = [dict] w/ params: values
        """
        _config = ConfigParser()
        _config.read(self.config_path)
        _config_dict = {s: dict(_config.items(s)) for s in _config.sections()}  # {'hparams': {'monitor': 'val_loss'}}
        _config_dict = {s: {k: self._convert(k, v) for k, v in subdict.items()} for s, subdict in _config_dict.items()}

        # combine sections 'dataset', 'model' & 'settings' to single section 'params'
        _config_dict['params'] = dict()
        for s in ['dataset', 'model', 'settings']:
            _config_dict['params'].update(_config_dict[s])
            _config_dict.pop(s)

        return _config, _config_dict

    def _convert(self, _input_key: str, _input_value):
        """
        convert _input string to str/int/float/bool
        -------------------------------------------
        :param _input_key:        [str],                e.g. 'lr_schedule' or 'prune_ratio_train' or 'checkpoints'
        :param _input_value:      [str],                e.g. 'constant'    or '0.01'              or 'False'
        :return: converted_input: [str/int/float/bool], e.g. 'constant'     or 0.01         or False
        """
        if _input_key in self.params.keys():
            convert_to = self.params[_input_key]
        elif _input_key in self.hparams.keys():
            convert_to = self.hparams[_input_key]
        else:
            raise Exception(f'_input_key = {_input_key} unknown.')

        if convert_to == 'str':
            return _input_value
        elif convert_to == 'int':
            return int(_input_value)
        elif convert_to == 'float':
            return float(_input_value)
        elif convert_to == 'bool':
            return _input_value not in ['False', 'false']
        else:
            raise Exception(f'convert_to = {convert_to} not known.')
