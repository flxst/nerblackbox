from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import TRAINING_DATASET_FRACTIONS


########################################################################################################################
# 3. APP.G.2 (Variants: Cool-down epochs)
########################################################################################################################
def experiments_variants_cooldown(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_3B_c{c}_x{x}_l{lr_cooldown_epochs}",
            "a": "adaptive",
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
            "lr_cooldown_epochs": lr_cooldown_epochs,
        }
        for c, x, lr_cooldown_epochs in product(["II", "III", "IV"], TRAINING_DATASET_FRACTIONS, [5, 9])
        if ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"3. EXPERIMENTS_VARIANTS_COOLDOWN: {len(exp)} = "
          f"3 x {len(TRAINING_DATASET_FRACTIONS)} x 2")
    return exp
