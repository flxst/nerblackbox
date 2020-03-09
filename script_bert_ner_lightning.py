
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

from utils.lightning_ner_model import LightningNerModel
from utils.env_variable import ENV_VARIABLE


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

    # logging & callbacks
    tb_logger = TensorBoardLogger(save_dir=log_dirs.tensorboard, name=params.experiment_run_name)
    checkpoint_callback = ModelCheckpoint(filepath=log_dirs.checkpoints) if params.checkpoints else None
    early_stopping_params = {k: vars(params)[k] for k in ['monitor', 'min_delta', 'patience', 'mode']}
    early_stop_callback = EarlyStopping(**early_stopping_params)

    # trainer
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=torch.cuda.device_count() if params.device.type == 'cuda' else None,
        use_amp=params.device.type == 'cuda',
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
    )

    trainer.fit(model)

    # logging
    _stopped_epoch_logging(tb_logger, hparams, early_stop_callback, model)


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


def _stopped_epoch_logging(_tb_logger,
                           _hparams,
                           _early_stop_callback,
                           _model,
                           metrics=('all_loss', 'fil_f1_micro', 'fil_f1_macro')):
    """
    log hparams and metrics for stopped epoch
    -----------------------------------------
    :param _tb_logger:           [pytorch lightning TensorboardLogger]
    :param _hparams:             [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param _early_stop_callback: [pytorch lightning callback]
    :param _model:               [LightningNerModel]
    :param metrics:              [tuple] of metrics to be logged in hparams section
    :return:
    """
    # stopped_epoch
    stopped_epoch = _early_stop_callback.stopped_epoch if _early_stop_callback.stopped_epoch else -1

    # hparams
    hparams_dict = {f'hparam/valid/{metric}': _model.epoch_valid_metrics[stopped_epoch][metric] for metric in metrics}
    hparams_dict['hparam/train/stopped_epoch'] = stopped_epoch
    _tb_logger.experiment.add_hparams(
        vars(_hparams),
        hparams_dict,
    )


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
    args_general.add_argument('--monitor', type=str, default='val_loss')                         # .. early stopping
    args_general.add_argument('--min_delta', type=float, default=0.00)                           # .. early stopping
    args_general.add_argument('--patience', type=int, default=2)                                 # .. early stopping
    args_general.add_argument('--mode', type=str, default='min')                                 # .. early stopping

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
