import pytest
from typing import Optional, Dict, Union

from nerblackbox.modules.main import NerBlackBoxMain
import os
from os.path import join, abspath, dirname
from pkg_resources import resource_filename

BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(BASE_DIR, "data")
os.environ["DATA_DIR"] = DATA_DIR
TRACKING_URI = resource_filename("nerblackbox", f"data/results/mlruns")


class TestMain:

    main = NerBlackBoxMain(flag="xyz", from_config=True)

    # 1 ################################################################################################################
    @pytest.mark.parametrize(
        "hparams, from_preset, hparams_processed",
        [
            (
                None,
                None,
                None,
            ),
            (
                {"multiple_runs": "2"},
                None,
                {"multiple_runs": "2"},
            ),
            (
                {"multiple_runs": "2"},
                "adaptive",
                {
                    "multiple_runs": "2",
                    "max_epochs": 250,
                    "early_stopping": True,
                    "lr_schedule": "constant",
                },
            ),
        ],
    )
    def test_process_hparams(
        self,
        hparams: Optional[Dict[str, Union[str, int, bool]]],
        from_preset: Optional[str],
        hparams_processed: Optional[Dict[str, str]],
    ):
        test_hparams_processed = self.main._process_hparams(hparams, from_preset)
        assert (
            test_hparams_processed == hparams_processed
        ), f"ERROR! test_hparams_processed = {test_hparams_processed} != {hparams_processed}"
