
from typing import Dict, Any
import logging
import warnings
from itertools import product

from nerblackbox import NerBlackBox

logging.basicConfig(
    level=logging.WARNING
)  # basic setting that is mainly applied to mlflow's default logging
warnings.filterwarnings("ignore")


########################################################################################################################
# GLOBAL SETTINGS ######################################################################################################
########################################################################################################################
GLOBAL_HYPERPARAMETERS = {
    "checkpoints": False,
    "multiple_runs": 5,
}

MODEL_DATASET = {
    "I": {
        "model": "bert-large-cased",
        "dataset": "conll2003"
    },
    "Ib": {
        "model": "bert-base-cased",
        "dataset": "conll2003"
    },
    "II": {
        "model": "distilbert-base-multilingual-cased",
        "dataset": "conll2003"
    },
    "III": {
        "model": "KB/bert-base-swedish-cased",
        "dataset": "swedish_ner_corpus"
    },
    "IV": {
        "model": "KB/bert-base-swedish-cased",
        "dataset": "swe_nerc"
    },
    "V": {
        "model": "TODO",
        "dataset": "TODO"
    },
}

FINE_TUNING_APPROACHES = ["original", "stable", "adaptive"]
# ANNOTATION_SCHEMES = ["bio", "bilou"]
DATASET_MODEL_COMBINATIONS = ["I", "II", "III", "IV", "V"]
TRAINING_DATASET_FRACTIONS = [0.005, 0.01, 0.015, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

########################################################################################################################
# 1. APP.D (Confirmation of existing results)
########################################################################################################################
EXPERIMENTS_CONFIRMATION = [
    {
        "experiment_name": f"exp_1_c{c}",
        "a": "original",
        "s": "bio",
        "c": c,
        "x": 1.00,
    }
    for c in ["Ib", "II", "IV"]
]
print(f"1. EXPERIMENTS_CONFIRMATION: {len(EXPERIMENTS_CONFIRMATION)}")

########################################################################################################################
# 2. APP.G.1 (Variants: Simplification)
########################################################################################################################
EXPERIMENTS_VARIANTS_SIMPLIFICATION = [
    {
        "experiment_name": f"exp_2_c{c}_x{x}_l{lr_cooldown_epochs}",
        "a": "no-training-resumption",
        "s": "bio",
        "c": c,
        "x": x,
        "lr_cooldown_epochs": lr_cooldown_epochs,
    }
    for c, x, lr_cooldown_epochs in product(["II", "III"], TRAINING_DATASET_FRACTIONS, [0, 3])
]
print(f"2. EXPERIMENTS_VARIANTS_SIMPLIFICATION: {len(EXPERIMENTS_VARIANTS_SIMPLIFICATION)} = "
      f"2 x {len(TRAINING_DATASET_FRACTIONS)} x 2")

########################################################################################################################
# 3. APP.G.2 (Variants: Cool-down epochs)
########################################################################################################################
EXPERIMENTS_VARIANTS_COOLDOWN = [
    {
        "experiment_name": f"exp_3_c{c}_x{x}_l{lr_cooldown_epochs}",
        "a": "adaptive",
        "s": "bio",
        "c": c,
        "x": x,
        "lr_cooldown_epochs": lr_cooldown_epochs,
    }
    for c, x, lr_cooldown_epochs in product(["II", "III"], TRAINING_DATASET_FRACTIONS, [5, 7])
]
print(f"3. EXPERIMENTS_VARIANTS_COOLDOWN: {len(EXPERIMENTS_VARIANTS_COOLDOWN)} = "
      f"2 x {len(TRAINING_DATASET_FRACTIONS)} x 2")

########################################################################################################################
# 4. APP.G.3 (Variants: Constant cool-down)
########################################################################################################################
EXPERIMENTS_VARIANTS_CONSTANT = [
    {
        "experiment_name": f"exp_4_c{c}_x{x}_p{patience}",
        "a": "adaptive",
        "s": "bio",
        "c": c,
        "x": x,
        "lr_cooldown_epochs": 0,
        "patience": patience,
    }
    for c, x, patience in product(["III"], TRAINING_DATASET_FRACTIONS, [3, 5, 7])
]
print(f"4. EXPERIMENTS_VARIANTS_CONSTANT: {len(EXPERIMENTS_VARIANTS_CONSTANT)} = "
      f"1 x {len(TRAINING_DATASET_FRACTIONS)} x 3")

########################################################################################################################
# 5. CH.5.1 + APP.E + APP.F (Main results)
########################################################################################################################
EXPERIMENTS_STANDARD = [
    {
        "experiment_name": f"exp_5_a{a}_c{c}_x{x}",
        "a": a,
        "s": "bio",
        "c": c,
        "x": x,
    }
    for a, c, x in product(FINE_TUNING_APPROACHES, DATASET_MODEL_COMBINATIONS, TRAINING_DATASET_FRACTIONS)
]
print(f"5. EXPERIMENTS_STANDARD: {len(EXPERIMENTS_STANDARD)} = "
      f"{len(FINE_TUNING_APPROACHES)} x {len(DATASET_MODEL_COMBINATIONS)} x {len(TRAINING_DATASET_FRACTIONS)}")

########################################################################################################################
# 6. APP.H (Ablation Studies)
########################################################################################################################
EXPERIMENTS_ABLATION_A = [
    {
        "experiment_name": f"exp_6A_a{a}_c{c}_x{x}",
        "a": a,
        "s": "bio",
        "c": c,
        "x": x,
        "max_epochs": 20,
    }
    for a, c, x in product(["stable", "hybrid"], ["II", "IV"], TRAINING_DATASET_FRACTIONS)
]
print(f"6A. EXPERIMENTS_ABLATION_A: {len(EXPERIMENTS_ABLATION_A)} = "
      f"2 x 2 x {len(TRAINING_DATASET_FRACTIONS)}")

MAX_EPOCH = {
    0.005: 11,
    0.01: 11,
    0.015: 11,
    0.02: 11,
    0.05: 11,
    0.10: 11,
    0.20: 11,
    0.40: 11,
    0.60: 11,
    0.80: 11,
    1.00: 11,
}  # TODO: use numbers from 6A
EXPERIMENTS_ABLATION_B = [
    {
        "experiment_name": f"exp_6B_a{a}_c{c}_x{x}",
        "a": a,
        "s": "bio",
        "c": c,
        "x": x,
        "max_epochs": MAX_EPOCH[x],
    }
    for a, c, x in product(["hybrid"], ["II", "IV"], TRAINING_DATASET_FRACTIONS)
]
print(f"6B. EXPERIMENTS_ABLATION_B: {len(EXPERIMENTS_ABLATION_B)} = "
      f"1 x 2 x {len(TRAINING_DATASET_FRACTIONS)}")

########################################################################################################################
# 7. CH.5.2 (Adaptive Fine-Tuning in Practice)
########################################################################################################################
EXPERIMENTS_PRACTICE_A = [
    {
        "experiment_name": f"exp_7A_c{c}_x{x}",
        "a": "adaptive",
        "s": "bio",
        "c": c,
        "x": x,
        "prune_ratio_val": x,
    }
    for c, x in product(["II", "IV"], TRAINING_DATASET_FRACTIONS)
]
print(f"7A. EXPERIMENTS_PRACTICE_A: {len(EXPERIMENTS_PRACTICE_A)} = "
      f"1 x 2 x {len(TRAINING_DATASET_FRACTIONS)}")

MAX_EPOCH = {
    0.005: 11,
    0.01: 11,
    0.015: 11,
    0.02: 11,
    0.05: 11,
    0.10: 11,
    0.20: 11,
    0.40: 11,
    0.60: 11,
    0.80: 11,
    1.00: 11,
}  # TODO: use numbers from 7A
EXPERIMENTS_PRACTICE_B = [
    {
        "experiment_name": f"exp_7B_c{c}_x{x}",
        "a": "hybrid",
        "s": "bio",
        "c": c,
        "x": x,
        "max_epochs": MAX_EPOCH[x],
        "prune_ratio_val": x,
        "train_on_val": True,
    }
    for c, x in product(["II", "IV"], TRAINING_DATASET_FRACTIONS)
]
print(f"7B. EXPERIMENTS_PRACTICE_B: {len(EXPERIMENTS_PRACTICE_B)} = "
      f"1 x 2 x {len(TRAINING_DATASET_FRACTIONS)}")

########################################################################################################################
# 8. CH.6 + APP.I (Annotation Schemes)
########################################################################################################################
EXPERIMENTS_SCHEME = [
    {
        "experiment_name": f"exp_8_a{a}_c{c}_x{x}",
        "a": a,
        "s": "bilou",
        "c": c,
        "x": x,
    }
    for a, c, x in product(FINE_TUNING_APPROACHES, ["II", "IV"], TRAINING_DATASET_FRACTIONS)
]
print(f"8. EXPERIMENTS_SCHEME: {len(EXPERIMENTS_SCHEME)} = "
      f"{len(FINE_TUNING_APPROACHES)} x 2 x {len(TRAINING_DATASET_FRACTIONS)}")

########################################################################################################################
# ALL
########################################################################################################################
EXPERIMENTS = \
    EXPERIMENTS_CONFIRMATION  # + \
    # EXPERIMENTS_VARIANTS_SIMPLIFICATION + \
    # EXPERIMENTS_VARIANTS_COOLDOWN + \
    # EXPERIMENTS_VARIANTS_CONSTANT + \
    # EXPERIMENTS_STANDARD + \
    # EXPERIMENTS_ABLATION_A + \
    # EXPERIMENTS_ABLATION_B + \
    # EXPERIMENTS_PRACTICE_A + \
    # EXPERIMENTS_PRACTICE_B + \
    # EXPERIMENTS_SCHEME
print(f"--> EXPERIMENTS: {len(EXPERIMENTS)}")


def process_experiment_dict(_experiment: Dict[str, Any]) -> Dict[str, Any]:
    _experiment["from_preset"] = _experiment["a"]
    _experiment["annotation_scheme"] = _experiment["s"]
    _experiment["prune_ratio_train"] = _experiment["x"]
    for field in ["model", "dataset"]:
        _experiment[field] = MODEL_DATASET[_experiment["c"]][field]
    _experiment.update(**GLOBAL_HYPERPARAMETERS)

    for removed_field in ["a", "s", "c", "x"]:
        _experiment.pop(removed_field)

    return _experiment


def main() -> None:

    nerbb = NerBlackBox()
    for experiment in EXPERIMENTS:
        experiment_name = experiment.pop("experiment_name")
        experiment = process_experiment_dict(experiment)
        print(experiment_name)
        print(experiment)
        # nerbb.run_experiment(experiment_name, **experiment)


if __name__ == "__main__":
    main()
