
from os.path import abspath, dirname
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))

ENV_VARIABLE = {
    'DIR_DATASETS': f'{BASE_DIR}/datasets',
    'DIR_EXPERIMENT_CONFIGS': f'{BASE_DIR}/experiment_configs',
    'DIR_CHECKPOINTS': f'{BASE_DIR}/results/checkpoints',
    'DIR_TENSORBOARD': f'{BASE_DIR}/results/tensorboard',
    'DIR_MLFLOW': f'{BASE_DIR}/results/mlruns',
    'LOG_FILE': f'{BASE_DIR}/results/logs.log',
    'MLFLOW_FILE': f'{BASE_DIR}/results/mlruns/mlflow_artifact.txt',
}
