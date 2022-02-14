from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import TRAINING_DATASET_FRACTIONS


########################################################################################################################
# 2. APP.G.1 (Variants: Simplification)
########################################################################################################################
def experiments_variants_simplification(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_2_c{c}_x{x}_l{lr_cooldown_epochs}",
            "a": "no-training-resumption",
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
            "lr_cooldown_epochs": lr_cooldown_epochs,
        }
        for c, x, lr_cooldown_epochs in product(["II", "III"], TRAINING_DATASET_FRACTIONS, [0, 7])
        if ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"2. EXPERIMENTS_VARIANTS_SIMPLIFICATION: {len(exp)} = "
          f"2 x {len(TRAINING_DATASET_FRACTIONS)} x 2")
    return exp
