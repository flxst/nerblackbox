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
            )
        ]
    )
    def test_process_kwargs_optional(self,
                                     kwargs_optional: Optional[Dict[str, Any]],
                                     kwargs: Dict[str, Any]):
        test_kwargs = self.nerbb._process_kwargs_optional(kwargs_optional)
        assert test_kwargs == kwargs, f"ERROR! test_kwargs = {test_kwargs} != {kwargs}"

    # 2 ################################################################################################################
    @pytest.mark.parametrize(
        "_kwargs, _hparams, _from_preset, _from_config",
        [
            (
                    {},
                    {},
                    None,
                    True,
            ),
            (
                    {"from_preset": None},
                    {},
                    None,
                    True,
            ),
            (
                    {"from_preset": "adaptive"},
                    {},
                    "adaptive",
                    False,
            ),
            (
                    {"a": 1, "from_preset": None},
                    {"a": 1},
                    None,
                    False,
            ),
            (
                    {"a": 1, "from_preset": "adaptive"},
                    {"a": 1},
                    "adaptive",
                    False,
            ),
        ]
    )
    def test_extract_hparams_and_from_preset(self,
                                             _kwargs: Dict[str, Any],
                                             _hparams: Dict[str, Any],
                                             _from_preset: str,
                                             _from_config: bool):
        test_hparams, test_from_preset, test_from_config = self.nerbb._extract_hparams_and_from_preset(_kwargs)
        assert test_hparams == _hparams, f"ERROR! test_hparams = {test_hparams} != {_hparams}"
        assert test_from_preset == _from_preset, f"ERROR! test_from_preset = {test_from_preset} != {_from_preset}"
        assert test_from_config == _from_config, f"ERROR! test_from_config = {test_from_config} != {_from_config}"
