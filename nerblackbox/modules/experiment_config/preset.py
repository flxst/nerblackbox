from typing import Dict, Optional, Union


PRESET: Dict[str, Dict[str, Union[str, int, bool]]] = {
    "adaptive": {
        "max_epochs": 250,
        "early_stopping": True,
        "lr_schedule": "constant",
    },
    "original": {
        "max_epochs": 5,
        "early_stopping": False,
        "lr_schedule": "linear",
    },
    "stable": {
        "max_epochs": 20,
        "early_stopping": False,
        "lr_schedule": "linear",
    },
    "hybrid": {
        # max epochs needs to be specified manually!
        "early_stopping": False,
        "lr_schedule": "hybrid",
    },
    ###################################
    # experimental
    ###################################
    "no-training-resumption": {
        "max_epochs": 100,
        "early_stopping": True,
        "lr_schedule": "constant",
        "lr_cooldown_restarts": False,
        # lr_cooldown_epochs needs to be specified manually! e.g. 0 or 7
    },
}


def get_preset(
    from_preset: Optional[str],
) -> Optional[Dict[str, Union[str, int, bool]]]:
    """
    Args:
        from_preset:     [str], e.g. 'adaptive' get experiment params & hparams from preset [HIERARCHY: II]

    Returns:
        preset: e.g. {'lr_schedule': 'constant', [..]}
    """
    if from_preset is None:
        preset = None
    else:
        assert (
            from_preset in PRESET.keys()
        ), f"ERROR! key = {from_preset} not in PRESET!"
        preset = PRESET[from_preset].copy()
    return preset
