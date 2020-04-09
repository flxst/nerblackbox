
import torch
import mlflow
from os.path import join
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from ner_black_box.ner_training.lightning_ner_model import LightningNerModel
from ner_black_box.ner_training.logging.default_logger import DefaultLogger
from ner_black_box.utils.util_functions import unify_parameters


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
    lightning_hparams = unify_parameters(params, hparams, log_dirs, experiment)

    tb_logger = logging_start(params, log_dirs)
    with mlflow.start_run(run_name=params.run_name, nested=experiment):

        model = LightningNerModel(lightning_hparams)
        callbacks = get_callbacks(params, hparams, log_dirs)

        trainer = Trainer(
            max_epochs=hparams.max_epochs,
            gpus=torch.cuda.device_count() if params.device.type == 'cuda' else None,
            precision=16 if (params.fp16 and params.device.type == 'cuda') else 32,
            amp_level='O1',
            logger=tb_logger,
            checkpoint_callback=callbacks['checkpoint'],
            early_stop_callback=callbacks['early_stop'],
        )
        trainer.fit(model)
        trainer.test()

        logging_end(tb_logger, hparams, callbacks, model, default_logger)


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
    _logger.log_info(f'> dataset_name:          {_params.dataset_name}')
    _logger.log_info(f'> dataset_tags:          {_params.dataset_tags}')
    _logger.log_info(f'> prune_ratio_train:     {_params.prune_ratio_train}')
    _logger.log_info(f'> prune_ratio_val:       {_params.prune_ratio_val}')
    _logger.log_info(f'> prune_ratio_test:      {_params.prune_ratio_test}')
    _logger.log_info('..')
    _logger.log_info(f'> pretrained_model_name: {_params.pretrained_model_name}')
    _logger.log_info(f'> uncased:               {_params.uncased}')
    _logger.log_info('..')
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
    model_checkpoint_filepath = join(_log_dirs.checkpoints, _params.experiment_run_name, '')

    _callbacks = {
        'checkpoint': ModelCheckpoint(filepath=model_checkpoint_filepath,
                                      verbose=True) if _params.checkpoints else None,
        'early_stop': EarlyStopping(**early_stopping_params, verbose=True)
    }
    return _callbacks


def logging_start(_params, _log_dirs):
    """
    :param _params:      [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _log_dirs:    [argparse.Namespace] attr: mlflow, tensorboard
    :return: _tb_logger: [pytorch lightning TensorBoardLogger]
    """
    # mlflow.tracking.set_tracking_uri(_log_dirs.mlflow)                # mlflow
    # mlflow.set_experiment(_params.experiment_name)                    # mlflow
    _tb_logger = TensorBoardLogger(save_dir=_log_dirs.tensorboard,    # tensorboard
                                   name=_params.experiment_run_name)
    return _tb_logger


def logging_end(_tb_logger, _hparams, _callbacks, _model, _logger):
    """
    :param _tb_logger:    [pytorch lightning TensorBoardLogger]
    :param _hparams:      [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param _callbacks:    [dict] w/ keys = 'checkpoint', 'early_stop' & values = [pytorch lightning callback]
    :param _model:        [LightningNerModel]
    :param _logger:       [DefaultLogger]
    :return: -
    """
    epoch_best = int(list(_callbacks['checkpoint'].best_k_models.keys())[0].split('epoch_')[-1].replace('.ckpt', ''))
    epoch_stopped = \
        _callbacks['early_stop'].stopped_epoch if _callbacks['early_stop'].stopped_epoch else _hparams.max_epochs-1

    _logger.log_info(f'epoch_best: {epoch_best}')
    _logger.log_info(f'epoch_stopped: {epoch_stopped}')

    _model.mlflow_client.log_metric('epoch_best', epoch_best)
    _model.mlflow_client.log_metric('epoch_stopped', epoch_stopped)

    _model.mlflow_client.finish_artifact_logger()                                       # mlflow
    _tb_logger_stopped_epoch(_tb_logger, _hparams, epoch_best, epoch_stopped, _model)   # tensorboard


def _tb_logger_stopped_epoch(_tb_logger,
                             _hparams,
                             _epoch_best,
                             _epoch_stopped,
                             _model,
                             metrics=('all_f1_micro', 'fil_f1_micro', 'chk_f1_micro')):
    """
    log hparams and metrics for stopped epoch
    -----------------------------------------
    :param _tb_logger:      [pytorch lightning TensorboardLogger]
    :param _hparams:        [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param _epoch_best:     [int]
    :param _epoch_stopped:  [int]
    :param _model:          [LightningNerModel]
    :param metrics:         [tuple] of metrics to be logged in hparams section
    :return:
    """
    hparams_dict = dict()

    # epoch_best & epoch_stopped
    hparams_dict['hparam/train/epoch_best'] = _epoch_best
    hparams_dict['hparam/train/epoch_stopped'] = _epoch_stopped

    # val/test
    hparams_val = {f'hparam/val/{metric.replace("+", "P")}': _model.epoch_metrics['val'][_epoch_stopped][metric]
                     for metric in metrics}
    hparams_test = {f'hparam/test/{metric.replace("+", "P")}': _model.epoch_metrics['test'][_epoch_stopped][metric]
                    for metric in metrics}
    hparams_dict.update(hparams_val)
    hparams_dict.update(hparams_test)

    """
    NOTE: 
    The following command (= that adds hyperparameters (_hparams) & most important metrics (hparams_dict) to tb logging)
    logs properly to tensorboard's SCALAR, but not to HPARAMS.
    Apparently it is incompatible with pytorch-lightning's default tensorboard logging, 
    see https://github.com/PyTorchLightning/pytorch-lightning/issues/1225
    So technically speaking there are 2 logs for each run, pytorch-lightning's default & the one below. 
    """
    # log
    _tb_logger.experiment.add_hparams(
        vars(_hparams),
        hparams_dict,
    )
