
from itertools import product
from typing import Optional, Dict, Union, Tuple, List
from nerblackbox.modules.utils.parameters import PARAMS, HPARAMS
from nerblackbox.modules.experiment_config.experiment_config import ExperimentConfig
from nerblackbox.modules.utils.util_functions import get_run_name, get_run_name_nr


class Experiment:

    def __init__(self,
                 experiment_name: str,
                 from_config: bool = True,
                 run_name: Optional[str] = None,
                 device: str = "gpu",
                 fp16: bool = True):
        """
        Args:
            experiment_name: e.g. 'exp1'
            from_config:     whether to read experiment config from file
            run_name:        e.g. 'runA'
            device:          e.g. 'gpu'
            fp16:
        """
        self.experiment_name = experiment_name
        self.from_config = from_config
        self.run_name = run_name
        self.device = device
        self.fp16 = fp16

        # def from_config(self):
            # pass

        self.exp_default = ExperimentConfig("default")
        if self.from_config:
            self.exp = ExperimentConfig(self.experiment_name)
        else:
            raise Exception(f"self.from_config = False currently not implemented!")

    def get_params_and_hparams(
            self, run_name_nr: Optional[str] = None
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        get dictionary of all parameters & their values that belong to
        either generally    to experiment (run_name == None)
        or     specifically to run        (run_name != None)

        Args:
            run_name_nr:        [str]  e.g. 'runA-1'

        Returns:
            params_and_hparams: [dict] e.g. {'patience': 2, 'mode': 'min', ..}
        """

        params_and_hparams = {}
        if run_name_nr is None:
            params_and_hparams.update(self.exp_default.config["params"])
            params_and_hparams.update(self.exp_default.config["hparams"])
            params_and_hparams.update(self.exp.config["params"])
            params_and_hparams.update(self.exp.config["hparams"])
        else:
            run_name = get_run_name(run_name_nr)
            params_and_hparams.update(self.exp.config[run_name])

        return params_and_hparams

    def parse(self) -> Tuple[List[str],
                             Dict[str, Dict[str, Union[str, int, float, bool]]],
                             Dict[str, Dict[str, Union[str, int, float, bool]]]]:
        """
        parse <experiment_name>.ini files
        if self.run_name is specified, parse only that run. else parse all runs.

        Returns:
            _runs_name_nr [list] of [str],
                                 e.g. ['runA-1', 'runA-2', 'runB-1', 'runB-2']
            _runs_params  [dict] w/ keys = run [str], values = params [dict],
                                 e.g. {'runA-1': {'patience': 2, 'mode': 'min', ..}, ..}
            _runs_hparams [dict] w/ keys = run [str], values = hparams [dict],
                                 e.g. {'runA-1': {'lr_max': 2e-5, 'max_epochs': 20, ..}, ..}
        """
        if (
                "params" in self.exp.config.keys()
                and "multiple_runs" in self.exp.config["params"].keys()
        ):
            multiple_runs = self.exp.config["params"]["multiple_runs"]
        elif (
                "params" in self.exp_default.config.keys()
                and "multiple_runs" in self.exp_default.config["params"].keys()
        ):
            multiple_runs = self.exp_default.config["params"]["multiple_runs"]
        else:
            raise Exception(
                f"multiple runs is neither specified in the experiment config nor in the default config."
            )

        # _params_config & _hparams_config
        _runs_params = dict()
        _runs_hparams = dict()

        if self.run_name is None:  # multiple runs
            run_names = self.exp.run_names
        else:
            run_names = [
                run_name for run_name in self.exp.run_names if run_name == self.run_name
            ]
            assert len(run_names) == 1

        _runs_name_nr = list()
        for run_name, run_nr in product(run_names, list(range(1, int(multiple_runs) + 1))):
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
            _run_params.update(self.exp_default.config["params"])
            _run_params.update(self.exp.config["params"])

            # _run_hparams
            _run_hparams = dict()
            _run_hparams.update(self.exp_default.config["hparams"])
            _run_hparams.update(self.exp.config["hparams"])

            for k, v in self.exp.config[run_name].items():
                if k in PARAMS:
                    _run_params[k] = v
                elif k in HPARAMS:
                    _run_hparams[k] = v
                else:
                    raise Exception(f"parameter = {k} is unknown.")

            _runs_name_nr.append(run_name_nr)
            _runs_params[run_name_nr] = _run_params
            _runs_hparams[run_name_nr] = _run_hparams

        assert set(_runs_params.keys()) == set(_runs_hparams.keys())

        return _runs_name_nr, _runs_params, _runs_hparams
