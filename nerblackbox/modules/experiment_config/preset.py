
from typing import Dict, Optional


PRESET = {
    "adaptive": {
        "max_epochs": 50,
        "early_stopping": True,
        "lr_schedule": "constant",
    },
    "original": {
        "max_epochs": 5,
        "early_stopping": False,
        "lr_schedule": "linear",
        "prune_ratio_val": 0.01,  # not needed as early_stopping is False
    },
    "stable": {
        "max_epochs": 20,
        "early_stopping": False,
        "lr_schedule": "linear",
        "prune_ratio_val": 0.01,  # not needed as early_stopping is False
    },
    "hybrid": {
        # max epochs needs to be specified manually!
        "early_stopping": False,
        "lr_schedule": "hybrid",
        "prune_ratio_val": 0.01,  # not needed as early_stopping is False
    },
    ###################################
    # experimental
    ###################################
    "no-training-resumption": {
        "max_epochs": 50,
        "early_stopping": True,
        "lr_schedule": "constant",
        "lr_cooldown_restarts": False,
        # lr_cooldown_epochs needs to be specified manually! e.g. 0 or 3
    },
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
