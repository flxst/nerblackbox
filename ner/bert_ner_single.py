
import torch
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from ner.lightning_ner_model import LightningNerModel
from ner.logging.default_logger import DefaultLogger


def main(params, hparams, log_dirs, experiment: bool):
    """
    :param params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param log_dirs:   [argparse.Namespace] attr: mlflow, tensorboard
    :param experiment: [bool] whether run is part of an experiment w/ multiple runs
    :return: -
    """
    default_logger = DefaultLogger(__file__, log_file=log_dirs.log_file, level=params.logging_level)  # python logging
    default_logger.clear()

    print_run_information(params, hparams, default_logger)

    tb_logger = logging_start(params, log_dirs)
    with mlflow.start_run(run_name=params.run_name, nested=experiment):

        model = LightningNerModel(params, hparams, log_dirs, experiment=experiment)
        callbacks = get_callbacks(params, hparams, log_dirs)

        trainer = Trainer(
            max_epochs=hparams.max_epochs,
            gpus=torch.cuda.device_count() if params.device.type == 'cuda' else None,
            use_amp=params.device.type == 'cuda',
            logger=tb_logger,
            checkpoint_callback=callbacks['checkpoint'],
            early_stop_callback=callbacks['early_stop'],
        )
        trainer.fit(model)
        trainer.test()

        logging_end(tb_logger, hparams, callbacks['early_stop'], model)


########################################################################################################################
# HELPER FUNCTIONS #####################################################################################################
########################################################################################################################
def print_run_information(_params, _hparams, _logger):
    """
    :param _params:   [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _hparams:  [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param _logger:   [DefaultLogger]
    :return: -
    """
    _logger.log_info('- PARAMS -----------------------------------------')
    _logger.log_info(f'> experiment_name: {_params.experiment_name}')
    _logger.log_info(f'> run_name:        {_params.run_name}')
    _logger.log_info('..')
    _logger.log_info(f'> available GPUs: {torch.cuda.device_count()}')
    _logger.log_info(f'> device:         {_params.device}')
    _logger.log_info(f'> fp16:           {_params.fp16}')
    _logger.log_info('..')
    _logger.log_info(f'> pretrained_model_name: {_params.pretrained_model_name}')
    _logger.log_info(f'> uncased:               {_params.uncased}')
    _logger.log_info(f'> dataset_name:          {_params.dataset_name}')
    _logger.log_info(f'> prune_ratio_train:     {_params.prune_ratio_train}')
    _logger.log_info(f'> prune_ratio_val:       {_params.prune_ratio_val}')
    _logger.log_info(f'> prune_ratio_test:      {_params.prune_ratio_test}')
    _logger.log_info(f'> checkpoints:           {_params.checkpoints}')
    _logger.log_info(f'> logging_level:         {_params.logging_level}')
    _logger.log_info('')
    _logger.log_info('- HPARAMS ----------------------------------------')
    _logger.log_info(f'> batch_size:       {_hparams.batch_size}')
    _logger.log_info(f'> max_seq_length:   {_hparams.max_seq_length}')
    _logger.log_info(f'> max_epochs:       {_hparams.max_epochs}')
    _logger.log_info(f'> monitor:          {_hparams.monitor}')
    _logger.log_info(f'> min_delta:        {_hparams.min_delta}')
    _logger.log_info(f'> patience:         {_hparams.patience}')
    _logger.log_info(f'> mode:             {_hparams.mode}')
    _logger.log_info(f'> lr_max:           {_hparams.lr_max}')
    _logger.log_info(f'> lr_warmup_epochs: {_hparams.lr_warmup_epochs}')
    _logger.log_info(f'> lr_schedule:      {_hparams.lr_schedule}')
    _logger.log_info(f'> lr_num_cycles:    {_hparams.lr_num_cycles}')
    _logger.log_info('')


def get_callbacks(_params, _hparams, _log_dirs):
    """
    :param _params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param _log_dirs:   [argparse.Namespace] attr: mlflow, tensorboard
    :return: _callbacks: [dict] w/ keys 'checkpoint', 'early_stop' & values = [pytorch lightning callback]
    """
    early_stopping_params = {k: vars(_hparams)[k] for k in ['monitor', 'min_delta', 'patience', 'mode']}
    _callbacks = {
        'checkpoint': ModelCheckpoint(filepath=_log_dirs.checkpoints) if _params.checkpoints else None,
        'early_stop': EarlyStopping(**early_stopping_params, verbose=True)
    }
    return _callbacks


def logging_start(_params, _log_dirs):
    """
    :param _params:      [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _log_dirs:    [argparse.Namespace] attr: mlflow, tensorboard
    :return: _tb_logger: [pytorch lightning TensorBoardLogger]
    """
    mlflow.tracking.set_tracking_uri(_log_dirs.mlflow)                # mlflow
    mlflow.set_experiment(_params.experiment_name)                    # mlflow
    _tb_logger = TensorBoardLogger(save_dir=_log_dirs.tensorboard,    # tensorboard
                                   name=_params.experiment_run_name)
    return _tb_logger


def logging_end(_tb_logger, _hparams, _callback_early_stop, _model):
    """
    :param _tb_logger:           [pytorch lightning TensorBoardLogger]
    :param _hparams:             [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param _callback_early_stop: [pytorch lightning EarlyStopping callback]
    :param _model:               [LightningNerModel]
    :return: -
    """
    _model.mlflow_client.finish_artifact_logger()                                  # mlflow
    _tb_logger_stopped_epoch(_tb_logger, _hparams, _callback_early_stop, _model)   # tensorboard


def _tb_logger_stopped_epoch(_tb_logger,
                             _hparams,
                             _early_stop_callback,
                             _model,
                             metrics=('all_f1_micro', 'fil_f1_micro', 'chk_f1_micro')):
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
    hparams_dict = dict()

    # stopped_epoch
    stopped_epoch = _early_stop_callback.stopped_epoch if _early_stop_callback.stopped_epoch else _hparams.max_epochs-1
    hparams_dict['hparam/train/stopped_epoch'] = stopped_epoch

    # val/test
    hparams_val = {f'hparam/val/{metric.replace("+", "P")}': _model.epoch_metrics['val'][stopped_epoch][metric]
                     for metric in metrics}
    hparams_test = {f'hparam/test/{metric.replace("+", "P")}': _model.epoch_metrics['test'][stopped_epoch][metric]
                    for metric in metrics}
    hparams_dict.update(hparams_val)
    hparams_dict.update(hparams_test)

    # log
    _tb_logger.experiment.add_hparams(
        vars(_hparams),
        hparams_dict,
    )
