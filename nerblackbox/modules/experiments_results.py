class ExperimentsResults:

    r"""class that contains results of multiple experiments.

        :param best_single_runs:  [pandas DataFrame] overview on best single runs
        :param best_average_runs: [pandas DataFrame] overview on best average runs
    """

    def __init__(self,
                 best_single_runs=None,
                 best_average_runs=None):

        self.best_single_runs = best_single_runs
        self.best_average_runs = best_average_runs
