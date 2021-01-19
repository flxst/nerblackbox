from typing import Optional, Any, Dict
from pandas import DataFrame


class ExperimentResults:

    r"""class that contains results of a single experiment.

    :param experiment:       [pandas DataFrame] overview on experiment parameters
    :param single_runs:      [pandas DataFrame] overview on run parameters & single  results
    :param average_runs:     [pandas DataFrame] overview on run parameters & average results
    :param best_single_run:  [dict] overview on best run parameters & single  results
    :param best_average_run: [dict] overview on best run parameters & average results
    :param best_model:       [NerModelPredict] instance
    """

    def __init__(
        self,
        experiment: Optional[DataFrame] = None,
        single_runs: Optional[DataFrame] = None,
        average_runs: Optional[DataFrame] = None,
        best_single_run: Optional[Dict] = None,
        best_average_run: Optional[Dict] = None,
        best_model: Optional[Any] = None,
    ):

        self.experiment = experiment
        self.single_runs = single_runs
        self.average_runs = average_runs
        self.best_single_run = best_single_run
        self.best_average_run = best_average_run
        self.best_model = best_model

    def _set_best_model(self, best_model):
        """set best model.

        :param best_model:       [NerModelPredict] instance
        """
        self.best_model = best_model
