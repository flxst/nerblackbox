from typing import Dict, Any, List, Optional, Union
from paper.experiments.global_variables import MODEL_DATASET, GLOBAL_HYPERPARAMETERS
from paper.experiments.experiments_appD import experiments_confirmation
from paper.experiments.experiments_appG1 import experiments_variants_simplification
from paper.experiments.experiments_appG2 import experiments_variants_cooldown
from paper.experiments.experiments_appG3 import experiments_variants_constant
from paper.experiments.experiments_appH import experiments_ablation_a, experiments_ablation_b
from paper.experiments.experiments_appK import experiments_practice_b
from paper.experiments.experiments_appL import experiments_practice_a, experiments_practice_c
from paper.experiments.experiments_main import experiments_main
from paper.experiments.experiments_scheme import experiments_scheme


########################################################################################################################
# ALL
########################################################################################################################
def get_experiments(exp: str, exp_filter: Dict[str, Optional[Union[str, float]]]) -> List[Dict[str, Any]]:
    if exp in ["appD", "confirmation"]:  # app.D
        return experiments_confirmation(exp_filter)
    elif exp in ["appG1", "variants_simplification"]:  # app.G
        return experiments_variants_simplification(exp_filter)
    elif exp in ["appG2", "variants_cooldown"]:  # app.G
        return experiments_variants_cooldown(exp_filter)
    elif exp in ["appG3", "variants_constant"]:  # app.G
        return experiments_variants_constant(exp_filter)
    elif exp in ["main", "standard"]:  # sec.5 & app.F & app.I & app. J
        return experiments_main(exp_filter)
    elif exp in ["appHa", "ablation_a"]:  # app.H
        return experiments_ablation_a(exp_filter)
    elif exp in ["appHb", "ablation_b"]:  # app.H
        return experiments_ablation_b(exp_filter)
    elif exp in ["appK", "practice_b"]:  # app.K
        return experiments_practice_b(exp_filter)
    elif exp in ["appLa", "practice_a"]:  # app.L
        return experiments_practice_a(exp_filter)
    elif exp in ["appLc", "practice_c"]:  # app.L
        return experiments_practice_c(exp_filter)
    elif exp == "scheme":  # additional
        return experiments_scheme(exp_filter)
    else:
        raise Exception(f"exp = {exp} unknown.")


def process_experiment_dict(_experiment: Dict[str, Any]) -> Dict[str, Any]:
    _experiment["from_preset"] = _experiment["a"]
    _experiment["annotation_scheme"] = _experiment["s"]
    _experiment["prune_ratio_train"] = _experiment["x"]
    for field in ["model", "dataset", "uncased"]:
        _experiment[field] = MODEL_DATASET[_experiment["c"]][field]
    _experiment.update(**GLOBAL_HYPERPARAMETERS)

    for removed_field in ["a", "s", "c", "x"]:
        _experiment.pop(removed_field)

    return _experiment
