from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import TRAINING_DATASET_FRACTIONS
from paper.experiments.experiments_main_results_epochs import MAX_EPOCH


########################################################################################################################
# 6. APP.H (Ablation Studies)
########################################################################################################################
def experiments_ablation_a(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_6A_a{a}_c{c}_x{x}",
            "a": a,
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
            "max_epochs": 20,
        }
        for a, c, x in product(["hybrid"], ["II", "III", "IV"], TRAINING_DATASET_FRACTIONS)
        if ("a" not in exp_filter or a == exp_filter["a"])
        and ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"6A. EXPERIMENTS_ABLATION_A: {len(exp)} = "
          f"1 x 3 x {len(TRAINING_DATASET_FRACTIONS)}")
    return exp


def experiments_ablation_b(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_6B_a{a}_c{c}_x{x}",
            "a": a,
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
            "max_epochs": MAX_EPOCH[c][x],
        }
        for a, c, x in product(["stable", "hybrid"], ["II", "III", "IV"], TRAINING_DATASET_FRACTIONS)
        if MAX_EPOCH[c][x] is not None
        and ("a" not in exp_filter or a == exp_filter["a"])
        and ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"6B. EXPERIMENTS_ABLATION_B: {len(exp)} = "
          f"2 x 3 x {len(TRAINING_DATASET_FRACTIONS)}")
    return exp
