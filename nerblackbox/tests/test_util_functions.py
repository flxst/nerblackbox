import pytest
from typing import List, Optional

from nerblackbox.modules.utils.util_functions import (
    checkpoint2epoch,
    epoch2checkpoint,
    get_run_name,
    get_run_name_nr,
    compute_mean_and_dmean,
)
from nerblackbox.tests.test_utils import pytest_approx


class TestUtilFunctions:
    @pytest.mark.parametrize(
        "checkpoint_name, epoch",
        [
            ("epoch=2.ckpt", 2),
            ("epoch=35.ckpt", 35),
        ],
    )
    def test_checkpoint2epoch(self, checkpoint_name: str, epoch: int):
        test_epoch = checkpoint2epoch(checkpoint_name)
        assert test_epoch == epoch, f"ERROR! test_epoch = {test_epoch} != {epoch}"

        test_checkpoint_name = epoch2checkpoint(epoch)
        assert (
            test_checkpoint_name == checkpoint_name
        ), f"ERROR! test_checkpoint_name = {test_checkpoint_name} != {checkpoint_name}"

    @pytest.mark.parametrize(
        "run_name, run_nr, run_name_nr",
        [
            ("runA", 3, "runA-3"),
        ],
    )
    def test_get_run_name(self, run_name: str, run_nr: int, run_name_nr: str):
        test_run_name = get_run_name(run_name_nr)
        assert (
            test_run_name == run_name
        ), f"ERROR! test_run_name = {test_run_name} != {run_name}"

        test_run_name_nr = get_run_name_nr(run_name, run_nr)
        assert (
            test_run_name_nr == run_name_nr
        ), f"ERROR! test_run_name_nr = {test_run_name_nr} != {run_name_nr}"

    @pytest.mark.parametrize(
        "list_of_float, " "mean, " "dmean",
        [
            ([], -1.0, None),
            ([1.0], 1.0, None),
            ([1.0, 1.0], 1.0, 0.0),
            ([1.0, 2.0], 1.5, 0.35355),  # = 0.5 / sqrt(2)
            (
                [0.7465069860279442, 0.7784431137724552],
                0.7624750499001998,
                0.0112911262464918,
            ),
        ],
    )
    def test_compute_mean_and_dmean(
        self,
        list_of_float: List[float],
        mean: float,
        dmean: Optional[float],
    ):
        test_mean, test_dmean = compute_mean_and_dmean(list_of_float)
        assert test_mean == pytest_approx(
            mean
        ), f"ERROR! test_mean = {test_mean} != {mean}"
        assert test_dmean == pytest_approx(
            dmean
        ), f"ERROR! test_dmean = {test_dmean} != {dmean}"
