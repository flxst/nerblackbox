GENERAL = {
    "experiment_name": "str",
    "from_config": "bool",
    "run_name": "str",
    "run_name_nr": "str",
    "device": "str",
    "fp16": "bool",
    "experiment_run_name_nr": "str",
}

DATASET = {
    "dataset_name": "str",
    "annotation_scheme": "str",
    "prune_ratio_train": "float",
    "prune_ratio_val": "float",
    "prune_ratio_test": "float",
    "train_on_val": "bool",
    "train_on_test": "bool",
}
MODEL = {
    "pretrained_model_name": "str",
}
SETTINGS = {
    "uncased": "bool",
    "checkpoints": "bool",
    "logging_level": "str",
    "multiple_runs": "int",
    "seed": "int",
}
PARAMS = {**DATASET, **MODEL, **SETTINGS}

HPARAMS = {
    "batch_size": "int",
    "max_seq_length": "int",
    "max_epochs": "int",
    "early_stopping": "bool",
    "monitor": "str",
    "min_delta": "float",
    "patience": "int",
    "mode": "str",
    "lr_max": "float",
    "lr_schedule": "str",
    "lr_warmup_epochs": "int",
    "lr_cooldown_epochs": "int",
    "lr_cooldown_restarts": "bool",
    "lr_num_cycles": "int",
}
LOG_DIRS = {
    "mlflow": "str",
    "tensorboard": "str",
    "checkpoints": "str",
    "log_file": "str",
    "mlflow_file": "str",
}
