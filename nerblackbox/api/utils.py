from typing import Optional, Dict, Any


class Utils:
    @classmethod
    def process_kwargs_optional(
        cls, _kwargs_optional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        general helper function
        filters out key-value pairs that have value = None

        Args:
            _kwargs_optional: e.g. {"a": 1, "b": None}

        Returns:
            _kwargs:          e.g. {"a": 1}
        """
        if _kwargs_optional is None:
            return {}
        else:
            return {k: v for k, v in _kwargs_optional.items() if v is not None}

    @classmethod
    def extract_hparams(cls, _kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            _kwargs: e.g. {"a": 1, "run_name": "runA-1"}

        Returns:
            _hparams: e.g. {"a": 1}
        """
        # hparams
        exclude_keys = ["training_name", "run_name", "device", "fp16"]
        _hparams = {
            _key: _kwargs[_key] for _key in _kwargs.keys() if _key not in exclude_keys
        }

        return _hparams
