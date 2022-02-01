from itertools import product
from typing import Optional, Dict, Union, List, Any
from nerblackbox.modules.utils.parameters import PARAMS, HPARAMS
from nerblackbox.modules.experiment_config.experiment_config import ExperimentConfig
from nerblackbox.modules.utils.util_functions import get_run_name_nr


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        from_config: bool = True,
        run_name: Optional[str] = None,
        device: str = "gpu",
        fp16: bool = True,
    ):
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

        self.exp_default = ExperimentConfig("default")
        self.exp = ExperimentConfig(self.experiment_name)

        # info for runs
        self.runs_name_nr: List[str] = list()
        self.runs_params: Dict[str, Dict[str, Union[str, int, float, bool]]] = dict()
        self.runs_hparams: Dict[str, Dict[str, Union[str, int, float, bool]]] = dict()
        self._create_info_for_runs()

        # info for mlflow
        self.params_and_hparams: Dict[str, Dict[str, Any]]
        self._create_info_for_mlflow()

    def _create_info_for_runs(self) -> None:
        """
        parse <experiment_name>.ini files
        if self.run_name is specified, parse only that run. else parse all runs.

        Created Attr:
            runs_name_nr [list] of [str],
                         e.g. ['runA-1', 'runA-2', 'runB-1', 'runB-2']
            runs_params  [dict] w/ keys = run [str], values = params [dict],
                         e.g. {'runA-1': {'patience': 2, 'mode': 'min', ..}, ..}
            runs_hparams [dict] w/ keys = run [str], values = hparams [dict],
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
        _runs_params: Dict[str, Union[str, int, float, bool]] = dict()
        _runs_hparams: Dict[str, Union[str, int, float, bool]] = dict()

        if self.run_name is None:  # multiple runs
            run_names = self.exp.run_names
        else:
            run_names = [
                run_name for run_name in self.exp.run_names if run_name == self.run_name
            ]
            assert len(run_names) == 1

        for run_name, run_nr in product(
            run_names, list(range(1, int(multiple_runs) + 1))
        ):
            run_name_nr = get_run_name_nr(run_name, run_nr)

            # _run_params
            _run_params: Dict[str, Union[str, int, float, bool]] = {
                "experiment_name": self.experiment_name,
                "from_config": self.from_config,
                "run_name": run_name,
                "run_name_nr": run_name_nr,
                "device": self.device,
                "fp16": self.fp16,
                "experiment_run_name_nr": f"{self.experiment_name}/{run_name_nr}",
            }
            _run_params.update(self.exp_default.config["params"])
            _run_params.update(self.exp.config["params"])

            # _run_hparams
            _run_hparams: Dict[str, Union[str, int, float, bool]] = dict()
            _run_hparams.update(self.exp_default.config["hparams"])
            _run_hparams.update(self.exp.config["hparams"])

            for k, v in self.exp.config[run_name].items():
                if k in PARAMS:
                    _run_params[k] = v
                elif k in HPARAMS:
                    _run_hparams[k] = v
                else:
                    raise Exception(f"parameter = {k} is unknown.")

            self.runs_name_nr.append(run_name_nr)
            self.runs_params[run_name_nr] = _run_params
            self.runs_hparams[run_name_nr] = _run_hparams

        assert set(self.runs_params.keys()) == set(self.runs_hparams.keys())

    def _create_info_for_mlflow(self) -> None:
        """
        get dictionary of all parameters & their values that belong to
        either generally    to experiment (key = "general")
        or     specifically to run        (key = run_name)

        Create Attr:
            params_and_hparams: [dict] e.g. {'patience': 2, 'mode': 'min', ..}

        Returns: -
        """

        self.params_and_hparams = {
            "general": {},
        }
        self.params_and_hparams["general"].update(self.exp_default.config["params"])
        self.params_and_hparams["general"].update(self.exp_default.config["hparams"])
        self.params_and_hparams["general"].update(self.exp.config["params"])
        self.params_and_hparams["general"].update(self.exp.config["hparams"])

        for run_name in self.exp.run_names:
            self.params_and_hparams[run_name] = self.exp.config[run_name]
