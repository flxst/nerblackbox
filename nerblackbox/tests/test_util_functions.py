import pytest
from typing import List, Optional

from nerblackbox.modules.utils.util_functions import compute_mean_and_dmean
from nerblackbox.tests.test_utils import pytest_approx


class TestUtilFunctions:

    @pytest.mark.parametrize(
        "list_of_float, " "mean, " "dmean",
        [
            ([1.], 1., None),
            ([1., 1.], 1., 0.),
            ([1., 2.], 1.5, 0.35355),  # = 0.5 / sqrt(2)
            ([0.7465069860279442, 0.7784431137724552], 0.7624750499001998, 0.0112911262464918)
        ]
    )
    def test_compute_mean_and_dmean(self,
                                    list_of_float: List[float],
                                    mean: float,
                                    dmean: Optional[float],
                                    ):
        test_mean, test_dmean = compute_mean_and_dmean(list_of_float)
        assert test_mean == pytest_approx(mean), f"ERROR! test_mean = {test_mean} != {mean}"
        assert test_dmean == pytest_approx(dmean), f"ERROR! test_dmean = {test_dmean} != {dmean}"

