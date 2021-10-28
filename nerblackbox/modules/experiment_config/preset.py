
from typing import Dict, Optional


PRESET = {
    "adaptive": {
        "model": "test_will_be_overwritten",
        "lr_schedule": "constant",
    }
}


def get_preset(from_preset: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Args:
        from_preset:     [str], e.g. 'adaptive' get experiment params & hparams from preset [HIERARCHY: II]

    Returns:
        preset: e.g. {'lr_schedule': 'constant', [..]}
    """
    if from_preset is None:
        return None
    else:
        assert from_preset in PRESET.keys(), f"ERROR! key = {from_preset} not in PRESET!"
        return PRESET[from_preset].copy()
