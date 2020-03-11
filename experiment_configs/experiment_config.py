
import os
from configparser import ConfigParser


class ExperimentConfig:
    """
    class that parses <experiment_name>.ini files
    """

    def __init__(self,
                 experiment_name: str,
                 run_name: str):
        """
        :param experiment_name: [str]
        :param run_name:        [str] or None
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.config_path = f'./experiment_configs/{self.experiment_name}.ini'
        if not os.path.isfile(self.config_path):
            raise Exception(f'config file at {self.config_path} does not exist')

    def parse(self):
        """
        parse <experiment_name>.ini files
        if self.run_name is specified, parse only that run. else parse all runs.
        ------------------------------------------------------------------------
        :return: runs             [list] of [str], e.g. ['run1', 'run2']
                 _params_configs  [dict] w/ keys = run_name [str], values = params [dict],
                                            e.g. {'run1': {'patience': 2, 'mode': 'min', ..}, ..}
                 _hparams_configs [dict] w/ keys = run_name [str], values = hparams [dict],
                                            e.g. {'run1': {'lr_max': 2e-5, 'max_epochs': 20, ..}, ..}
        """
        config = ConfigParser()
        config.read(self.config_path)

        config_dict = {s: dict(config.items(s)) for s in config.sections()}
        config_dict = {s: {k: self._convert(v) for k, v in subdict.items()} for s, subdict in config_dict.items()}

        # _params_config & _hparams_config
        _params_configs = dict()
        _hparams_configs = dict()

        if self.run_name is None:  # multiple runs
            runs = [run for run in config.sections() if run not in ['params', 'hparams']]
        else:
            runs = [run for run in config.sections() if run == self.run_name]
            assert len(runs) == 1

        for run in runs:
            _params_configs[run] = config_dict['params'].copy()
            _params_configs[run]['run_name'] = run
            _params_configs[run]['experiment_run_name'] = f'{self.experiment_name}/{run}'

            _hparams_configs[run] = config_dict['hparams'].copy()
            _hparams_configs[run].update(config_dict[run])

        assert set(_params_configs.keys()) == set(_hparams_configs.keys())

        return runs, _params_configs, _hparams_configs

    @staticmethod
    def _convert(_input: str):
        """
        convert _input string to str/int/float/bool
        -------------------------------------------
        :param _input:            [str],                e.g. 'str|constant' or 'float|0.01' or 'bool|False'
        :return: converted_input: [str/int/float/bool], e.g. 'constant'     or 0.01         or False
        """
        convert_to, element = _input.split('|')
        if convert_to == 'str':
            return element
        elif convert_to == 'int':
            return int(element)
        elif convert_to == 'float':
            return float(element)
        elif convert_to == 'bool':
            return element not in ['False', 'false']
        else:
            raise Exception(f'convert_to = {convert_to} not known.')

