from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import TRAINING_DATASET_FRACTIONS


########################################################################################################################
# 4. APP.G.3 (Variants: Constant cool-down)
########################################################################################################################
def experiments_variants_constant(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_4_c{c}_x{x}_p{patience}",
            "a": "adaptive",
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
            "lr_cooldown_epochs": 0,
            "patience": patience,
        }
        for c, x, patience in product(["II", "III"], TRAINING_DATASET_FRACTIONS, [5, 7, 9])
        if ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"4. EXPERIMENTS_VARIANTS_CONSTANT: {len(exp)} = "
          f"2 x {len(TRAINING_DATASET_FRACTIONS)} x 3")
    return exp
