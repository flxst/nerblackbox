from typing import Optional
from pandas import DataFrame


class ExperimentsResults:

    """class that contains results of multiple experiments."""

    def __init__(
        self,
        best_single_runs: Optional[DataFrame] = None,
        best_average_runs: Optional[DataFrame] = None,
    ):
        """
        Args:
            best_single_runs: overview on best single runs
            best_average_runs: overview on best average runs
        """

        self.best_single_runs = best_single_runs
        self.best_average_runs = best_average_runs
