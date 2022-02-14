from typing import Optional, Dict, List, Any


########################################################################################################################
# 1. APP.D (Confirmation of existing results)
########################################################################################################################
def experiments_confirmation(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_1_c{c}",
            "a": "original",
            "s": "bio",
            "c": c,
            "x": 1.00,  # same as default
            "prune_ratio_val": 1.00,  # same as default
            "train_on_val": True if c == "IV" else False,
        }
        for c in ["Ib", "II", "IV"]
        if ("c" not in exp_filter or c == exp_filter["c"])
    ]
    print(f"1. EXPERIMENTS_CONFIRMATION: {len(exp)}")
    return exp
