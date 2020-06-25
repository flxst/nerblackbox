
import os


def env_variable(key):
    data_path = os.environ.get('DATA_DIR')
    
    env_variable_dict = {
        'DIR_DATASETS': f'{data_path}/datasets',
        'DIR_EXPERIMENT_CONFIGS': f'{data_path}/experiment_configs',
        'DIR_RESULTS': f'{data_path}/results',
        'DIR_CHECKPOINTS': f'{data_path}/results/checkpoints',
        'DIR_TENSORBOARD': f'{data_path}/results/tensorboard',
        'DIR_MLFLOW': f'{data_path}/results/mlruns',
        'LOG_FILE': f'{data_path}/results/logs.log',
        'MLFLOW_FILE': f'{data_path}/results/mlruns/mlflow_artifact.txt',
    }
    
    return env_variable_dict[key]
