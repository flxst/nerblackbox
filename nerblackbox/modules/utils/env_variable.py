import os


def env_variable(key: str) -> str:
    data_dir = os.environ.get("DATA_DIR")
    assert data_dir is not None, "ERROR! DATA_DIR not found."

    env_variable_dict = {
        "DATA_DIR": data_dir,
        "DIR_DATASETS": f"{data_dir}/datasets",
        "DIR_EXPERIMENT_CONFIGS": f"{data_dir}/experiment_configs",
        "DIR_RESULTS": f"{data_dir}/results",
        "DIR_CHECKPOINTS": f"{data_dir}/results/checkpoints",
        "DIR_TENSORBOARD": f"{data_dir}/results/tensorboard",
        "DIR_MLFLOW": f"{data_dir}/results/mlruns",
        "LOG_FILE": f"{data_dir}/results/logs.log",
        "MLFLOW_FILE": f"{data_dir}/results/mlruns/mlflow_artifact.txt",
    }

    return env_variable_dict[key]
