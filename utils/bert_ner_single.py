
import torch
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from utils.lightning_ner_model import LightningNerModel
from utils.default_logger import DefaultLogger


def main(params, hparams, log_dirs, experiment):
    """
    :param params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param log_dirs:   [argparse.Namespace] attr: mlflow, tensorboard
    :param experiment: [bool] whether run is part of an experiment w/ multiple runs
    :return: -
    """
    DefaultLogger.clear()
    _print_run_information(params, hparams)

    # mlflow start
    mlflow.tracking.set_tracking_uri(log_dirs.mlflow)
    mlflow.set_experiment(params.experiment_name)
    with mlflow.start_run(run_name=params.run_name, nested=experiment):

        # model
        model = LightningNerModel(params, hparams, log_dirs, experiment=experiment)

        # logging & callbacks
        tb_logger = TensorBoardLogger(save_dir=log_dirs.tensorboard, name=params.experiment_run_name)
        checkpoint_callback = ModelCheckpoint(filepath=log_dirs.checkpoints) if params.checkpoints else None
        early_stopping_params = {k: vars(hparams)[k] for k in ['monitor', 'min_delta', 'patience', 'mode']}
        early_stop_callback = EarlyStopping(**early_stopping_params, verbose=True)

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
        _tb_logger_stopped_epoch(tb_logger, hparams, early_stop_callback, model)


def _print_run_information(_params, _hparams):
    """
    :param _params:   [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _hparams:  [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :return: -
    """
    default_logger = DefaultLogger(__file__, level=_params.logging_level)  # python logging

    default_logger.log_info('- PARAMS -----------------------------------------')
    default_logger.log_info(f'> experiment_name: {_params.experiment_name}')
    default_logger.log_info(f'> run_name:        {_params.run_name}')
    default_logger.log_info('..')
    default_logger.log_info(f'> available GPUs: {torch.cuda.device_count()}')
    default_logger.log_info(f'> device:         {_params.device}')
    default_logger.log_info(f'> fp16:           {_params.fp16}')
    default_logger.log_info('..')
    default_logger.log_info(f'> pretrained_model_name: {_params.pretrained_model_name}')
    default_logger.log_info(f'> uncased:               {_params.uncased}')
    default_logger.log_info(f'> dataset_name:          {_params.dataset_name}')
    default_logger.log_info(f'> prune_ratio_train:     {_params.prune_ratio_train}')
    default_logger.log_info(f'> prune_ratio_valid:     {_params.prune_ratio_valid}')
    default_logger.log_info(f'> checkpoints:           {_params.checkpoints}')
    default_logger.log_info(f'> logging_level:         {_params.logging_level}')
    default_logger.log_info('')
    default_logger.log_info('- HPARAMS ----------------------------------------')
    default_logger.log_info(f'> batch_size:       {_hparams.batch_size}')
    default_logger.log_info(f'> max_seq_length:   {_hparams.max_seq_length}')
    default_logger.log_info(f'> max_epochs:       {_hparams.max_epochs}')
    default_logger.log_info(f'> monitor:          {_hparams.monitor}')
    default_logger.log_info(f'> min_delta:        {_hparams.min_delta}')
    default_logger.log_info(f'> patience:         {_hparams.patience}')
    default_logger.log_info(f'> mode:             {_hparams.mode}')
    default_logger.log_info(f'> lr_max:           {_hparams.lr_max}')
    default_logger.log_info(f'> lr_warmup_epochs: {_hparams.lr_warmup_epochs}')
    default_logger.log_info(f'> lr_schedule:      {_hparams.lr_schedule}')
    default_logger.log_info(f'> lr_num_cycles:    {_hparams.lr_num_cycles}')
    default_logger.log_info('')


def _tb_logger_stopped_epoch(_tb_logger,
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
    stopped_epoch = _early_stop_callback.stopped_epoch if _early_stop_callback.stopped_epoch else _hparams.max_epochs-1

    # hparams
    hparams_dict = {f'hparam/valid/{metric}': _model.epoch_valid_metrics[stopped_epoch][metric] for metric in metrics}
    hparams_dict['hparam/train/stopped_epoch'] = stopped_epoch
    _tb_logger.experiment.add_hparams(
        vars(_hparams),
        hparams_dict,
    )
