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
        "dataset": "conll2003",
        "uncased": False,
    },
    "Ib": {
        "model": "bert-base-cased",
        "dataset": "conll2003",
        "uncased": False,
    },
    "II": {
        "model": "distilbert-base-multilingual-cased",
        "dataset": "conll2003",
        "uncased": False,
    },
    "III": {
        "model": "KB/bert-base-swedish-cased",
        "dataset": "swedish_ner_corpus",
        "uncased": False,
    },
    "IV": {
        "model": "KB/bert-base-swedish-cased",
        "dataset": "swe_nerc",
        "uncased": False,
    },
    "V": {
        "model": "mrm8488/electricidad-base-discriminator",
        "dataset": "ehealth_kd",
        "uncased": True,
    },
}

FINE_TUNING_APPROACHES = ["original", "stable", "adaptive"]
DATASET_MODEL_COMBINATIONS = ["I", "II", "III", "IV", "V"]
TRAINING_DATASET_FRACTIONS = [0.005, 0.01, 0.015, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
