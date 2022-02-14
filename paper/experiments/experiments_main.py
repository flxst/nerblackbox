from typing import Optional, Dict, List, Any
from itertools import product
from paper.experiments.global_variables import \
    FINE_TUNING_APPROACHES, DATASET_MODEL_COMBINATIONS, TRAINING_DATASET_FRACTIONS


########################################################################################################################
# 5. SEC.5 + APP.F + App.I + App. J (Main results)
########################################################################################################################
def experiments_main(exp_filter: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    exp = [
        {
            "experiment_name": f"exp_5_a{a}_c{c}_x{x}",
            "a": a,
            "s": "bio",
            "c": c,
            "x": x,
            "prune_ratio_val": x,
        }
        for a, c, x in product(FINE_TUNING_APPROACHES, DATASET_MODEL_COMBINATIONS, TRAINING_DATASET_FRACTIONS)
        if ("a" not in exp_filter or a == exp_filter["a"])
        and ("c" not in exp_filter or c == exp_filter["c"])
        and ("x" not in exp_filter or x == exp_filter["x"])
    ]
    print(f"5. EXPERIMENTS_MAIN: {len(exp)} = "
          f"{len(FINE_TUNING_APPROACHES)} x {len(DATASET_MODEL_COMBINATIONS)} x {len(TRAINING_DATASET_FRACTIONS)}")
    return exp
