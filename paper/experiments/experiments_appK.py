from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import TRAINING_DATASET_FRACTIONS


########################################################################################################################
# 7. APP.K (Dependency of the Model Performance on the Validation Dataset Size)
########################################################################################################################
def experiments_practice_b(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_7B_a{a}_c{c}_x{x}",
            "a": a,
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": xval,
        }
        for a, c, x, xval in product(["adaptive"], ["II", "III", "IV"], TRAINING_DATASET_FRACTIONS, [0.01, 0.1, 1.0])
        if ("a" not in exp_filter or a == exp_filter["a"])
        and ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
        and ("xval" not in exp_filter or xval == exp_filter["xval"])
    ]
    print(f"7B. EXPERIMENTS_PRACTICE_B: {len(exp)} = "
          f"1 x 3 x {len(TRAINING_DATASET_FRACTIONS)} x 3")
    return exp
