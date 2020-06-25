
class ExperimentResults:

    r"""class that contains results of a single experiment.

        :param experiment:       [pandas DataFrame] overview on experiment parameters
        :param single_runs:      [pandas DataFrame] overview on run parameters & single  results
        :param average_runs:     [pandas DataFrame] overview on run parameters & average results
        :param best_single_run:  [dict] overview on best run parameters & single  results
        :param best_average_run: [dict] overview on best run parameters & average results
        :param best_model:       [LightningNerModelPredict] instance
    """

    def __init__(self,
                 experiment=None,
                 single_runs=None,
                 average_runs=None,
                 best_single_run=None,
                 best_average_run=None,
                 best_model=None):

        self.experiment = experiment
        self.single_runs = single_runs
        self.average_runs = average_runs
        self.best_single_run = best_single_run
        self.best_average_run = best_average_run
        self.best_model = best_model

    def _set_best_model(self,
                        best_model):
        """set best model.

        :param best_model:       [LightningNerModelPredict] instance
        """
        self.best_model = best_model
