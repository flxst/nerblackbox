from typing import Optional, Dict
from pandas import DataFrame
from nerblackbox.modules.ner_training.ner_model_predict import NerModelPredict


class ExperimentResults:

    """class that contains results of a single experiment."""

    def __init__(
        self,
        experiment: Optional[DataFrame] = None,
        single_runs: Optional[DataFrame] = None,
        average_runs: Optional[DataFrame] = None,
        best_single_run: Optional[Dict] = None,
        best_average_run: Optional[Dict] = None,
        best_model: Optional[NerModelPredict] = None,
    ):
        """

        Args:
            experiment: overview on experiment parameters
            single_runs: overview on run parameters & single  results
            average_runs: overview on run parameters & average results
            best_single_run: overview on best run parameters & single results
            best_average_run: overview on best run parameters & average results
            best_model: best model
        """

        self.experiment = experiment
        self.single_runs = single_runs
        self.average_runs = average_runs
        self.best_single_run = best_single_run
        self.best_average_run = best_average_run
        self.best_model = best_model

    def _set_best_model(self, best_model: NerModelPredict) -> None:
        """set best model.

        Args:
            best_model: best model
        """
        self.best_model = best_model
