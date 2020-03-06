
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

from utils.lightning_ner_model import LightningNerModel
from utils.env_variable import ENV_VARIABLE


def _print_device_information(_params):
    """
    :param _params:   [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :return: -
    """
    print('---------------------------')
    print(f'> Available GPUs: {torch.cuda.device_count()}')
    print(f'> Using device:   {_params.device}')
    print(f'> Using fp16:     {_params.fp16}')
    print('---------------------------')


def main(params, hparams, log_dirs):
    """
    :param params:   [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param hparams:  [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param log_dirs: [argparse.Namespace] attr: mlflow, tensorboard
    :return: -
    """
    _print_device_information(params)

    # model
    model = LightningNerModel(params, hparams, log_dirs)

    # logging & checkpoints
    tb_logger = TensorBoardLogger(save_dir=log_dirs.tensorboard, name=params.experiment_run_name)
    checkpoint_callback = ModelCheckpoint(filepath=log_dirs.checkpoints) if params.checkpoints else None

    # trainer
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=torch.cuda.device_count() if params.device.type == 'cuda' else None,
        use_amp=params.device.type == 'cuda',
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params
    args_general = parser.add_argument_group('args_general')
    args_general.add_argument('--experiment_name', type=str, default='Default')  # .. logging w/ mlflow & tensorboard
    args_general.add_argument('--run_name', type=str, default='Default')         # .. logging w/ mlflow & tensorboard
    args_general.add_argument('--pretrained_model_name', type=str, default='af-ai-center/bert-base-swedish-uncased')
    args_general.add_argument('--dataset_name', type=str, default='swedish_ner_corpus')          # .. model & dataset
    args_general.add_argument('--device', type=str, default='gpu')                               # .. device
    args_general.add_argument('--fp16', type=bool, default=False)                                # .. device
    args_general.add_argument('--checkpoints', action='store_true')                              # .. checkpoints

    # hparams
    args_hparams = parser.add_argument_group('args_hparams')
    args_hparams.add_argument('--batch_size', type=int, default=16)
    args_hparams.add_argument('--max_seq_length', type=int, default=64)
    args_hparams.add_argument('--max_epochs', type=int, default=2)
    args_hparams.add_argument('--prune_ratio_train', type=float, default=0.01)
    args_hparams.add_argument('--prune_ratio_valid', type=float, default=0.01)
    args_hparams.add_argument('--lr_max', type=float, default=2e-5)
    args_hparams.add_argument('--lr_warmup_epochs', type=int, default=1)
    args_hparams.add_argument('--lr_schedule', type=str, default='constant')
    args_hparams.add_argument('--lr_num_cycles', type=float, default=4)

    # parsing
    _args = parser.parse_args()
    _params, _hparams = None, None
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(_args, a.dest, None) for a in group._group_actions}
        if group.title == 'args_general':
            group_dict['experiment_run_name'] = group_dict['experiment_name'] + '/' + group_dict['run_name']
            group_dict['device'] = torch.device('cuda' if torch.cuda.is_available() and group_dict['device'] == 'gpu' else 'cpu')
            group_dict['fp16'] = True if group_dict['fp16'] and group_dict['device'] == 'cuda' else False
            _params = argparse.Namespace(**group_dict)
        elif group.title == 'args_hparams':
            _hparams = argparse.Namespace(**group_dict)

    # log_dirs
    _log_dirs_dict = {
        'mlflow': ENV_VARIABLE['DIR_MLFLOW'],
        'tensorboard': ENV_VARIABLE['DIR_TENSORBOARD'],
        'checkpoints': ENV_VARIABLE['DIR_CHECKPOINTS'],
    }
    _log_dirs = argparse.Namespace(**_log_dirs_dict)

    # main
    main(_params, _hparams, _log_dirs)
