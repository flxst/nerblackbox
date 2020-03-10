from os.path import abspath, dirname
BASE_DIR = abspath(dirname(dirname(__file__)))

DOWNSTREAM_TASK = 'ner'

ENV_VARIABLE = {
    'DIR_DATASETS': f'{BASE_DIR}/datasets/{DOWNSTREAM_TASK}',
    'DIR_CHECKPOINTS': f'{BASE_DIR}/results/checkpoints',
    'DIR_TENSORBOARD': f'{BASE_DIR}/results/tensorboard',
    'DIR_MLFLOW': f'{BASE_DIR}/results/mlruns',
}
