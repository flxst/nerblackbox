from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import TRAINING_DATASET_FRACTIONS
from paper.experiments.experiments_main_results_epochs import MAX_EPOCH


########################################################################################################################
# App.L (Adaptive Fine-Tuning in Practice: Using the Validation Set for Training)
########################################################################################################################
def experiments_practice_a(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_7A_a{a}_c{c}_x{x}",
            "a": a,
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
            "train_on_val": True,
        }
        for a, c, x in product(["original", "stable"], ["II", "III", "IV"], TRAINING_DATASET_FRACTIONS)
        if ("a" not in exp_filter or a == exp_filter["a"])
        and ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"7A. EXPERIMENTS_PRACTICE_A: {len(exp)} = "
          f"2 x 3 x {len(TRAINING_DATASET_FRACTIONS)}")
    return exp


def experiments_practice_c(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_7C_a{a}_c{c}_x{x}",
            "a": a,
            "s": "bio",
            "c": c,
            "x": x,
            "max_epochs": MAX_EPOCH[c][x],
            "prune_ratio_val": x,
            "train_on_val": True,
        }
        for a, c, x in product(["hybrid"], ["II", "III", "IV"], TRAINING_DATASET_FRACTIONS)
        if MAX_EPOCH[c][x] is not None
        and ("a" not in exp_filter or a == exp_filter["a"])
        and ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"7C. EXPERIMENTS_PRACTICE_C: {len(exp)} = "
          f"1 x 3 x {len(TRAINING_DATASET_FRACTIONS)}")
    return exp
