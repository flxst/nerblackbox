import pytest
from typing import Optional, Dict, Any

from nerblackbox.api import NerBlackBox


class TestApi:

    nerbb = NerBlackBox()

    # 1 ################################################################################################################
    @pytest.mark.parametrize(
        "kwargs_optional, kwargs",
        [
            (
                None,
                {},
            ),
            (
                {"a": 1, "b": None},
                {"a": 1},
            ),
        ],
    )
    def test_process_kwargs_optional(
        self, kwargs_optional: Optional[Dict[str, Any]], kwargs: Dict[str, Any]
    ):
        test_kwargs = self.nerbb._process_kwargs_optional(kwargs_optional)
        assert test_kwargs == kwargs, f"ERROR! test_kwargs = {test_kwargs} != {kwargs}"

    # 2 ################################################################################################################
    @pytest.mark.parametrize(
        "_kwargs, _hparams",
        [
            (
                {},
                {},
            ),
            (
                {"fp16": True},
                {},
            ),
            (
                {"a": 1, "fp16": None},
                {"a": 1},
            ),
        ],
    )
    def test_extract_hparams(self, _kwargs: Dict[str, Any], _hparams: Dict[str, Any]):
        test_hparams = self.nerbb._extract_hparams(_kwargs)
        assert (
            test_hparams == _hparams
        ), f"ERROR! test_hparams = {test_hparams} != {_hparams}"
