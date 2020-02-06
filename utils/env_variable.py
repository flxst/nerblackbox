DOWNSTREAM_TASK = 'ner'

ENV_VARIABLE = {
    'DIR_PRETRAINED_MODELS': './pretrained_models',  # TODO: get rid of this line
    'DIR_DATASETS': f'./datasets/{DOWNSTREAM_TASK}',
    'DIR_CHECKPOINTS': f'./checkpoints/{DOWNSTREAM_TASK}',
}


