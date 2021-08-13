import torch
import mlflow
import os
from os.path import join
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from nerblackbox.modules.ner_training.ner_model_train import (
    NerModelTrain,
)
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger
from nerblackbox.modules.utils.util_functions import unify_parameters
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.utils.util_functions import (
    get_package_version,
    checkpoint2epoch,
    epoch2checkpoint,
)


def execute_single_run(params, hparams, log_dirs, experiment: bool):
    """
    :param params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param log_dirs:   [argparse.Namespace] attr: mlflow, tensorboard
    :param experiment: [bool] whether run is part of an experiment w/ multiple runs
    :return: -
    """
    # seed
    seed = params.seed + int(params.run_name_nr.split("-")[1])  # run_name_nr = e.g. 'runA-1', 'runA-2', 'runB-1', ..
    seed_everything(seed)

    default_logger = DefaultLogger(
        __file__, log_file=log_dirs.log_file, level=params.logging_level
    )  # python logging
    default_logger.clear()

    print_run_information(params, hparams, default_logger, seed)

    lightning_hparams = unify_parameters(params, hparams, log_dirs, experiment)

    tb_logger = logging_start(params, log_dirs)
    with mlflow.start_run(run_name=params.run_name_nr, nested=experiment):

        model = NerModelTrain(lightning_hparams)
        callbacks = get_callbacks(params, hparams, log_dirs)

        trainer = Trainer(
            max_epochs=hparams.max_epochs,
            gpus=torch.cuda.device_count() if params.device.type == "cuda" else None,
            precision=16 if (params.fp16 and params.device.type == "cuda") else 32,
            amp_level="O1",
            logger=tb_logger,
            callbacks=callbacks,
        )
        trainer.fit(model)
        callback_info = get_callback_info(callbacks, params, hparams)

        default_logger.log_info(
            "--- LOAD BEST CHECKPOINT FOR TESTING AND DETAILED RESULTS ---"
        )
        model_best = NerModelTrain.load_from_checkpoint(
            checkpoint_path=callback_info["checkpoint_best"]
        )
        trainer_best = Trainer(
            gpus=torch.cuda.device_count() if params.device.type == "cuda" else None,
            precision=16 if (params.fp16 and params.device.type == "cuda") else 32,
            logger=tb_logger,
        )
        trainer_best.test(model_best)

        # logging end
        logging_end(
            tb_logger, callback_info, hparams, model, model_best, default_logger
        )

        # remove checkpoint
        if params.checkpoints is False:
            remove_checkpoint(callback_info["checkpoint_best"], default_logger)


########################################################################################################################
# HELPER FUNCTIONS #####################################################################################################
########################################################################################################################
def print_run_information(_params, _hparams, _logger, _seed: int):
    """
    :param _params:   [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _hparams:  [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param _logger:   [DefaultLogger]
    :param _seed:     [int]
    :return: -
    """
    _logger.log_info(f">>> NER BLACK BOX VERSION: {get_package_version()}")
    _logger.log_info("- PARAMS -----------------------------------------")
    _logger.log_info(f"> experiment_name: {_params.experiment_name}")
    _logger.log_info(f"> run_name_nr:     {_params.run_name_nr}")
    _logger.log_info("..")
    _logger.log_info(f"> available GPUs: {torch.cuda.device_count()}")
    _logger.log_info(f"> device:         {_params.device}")
    _logger.log_info(f"> fp16:           {_params.fp16}")
    _logger.log_info("..")
    _logger.log_info(f"> dataset_name:          {_params.dataset_name}")
    _logger.log_info(f"> annotation_scheme:     {_params.annotation_scheme}")
    _logger.log_info(f"> prune_ratio_train:     {_params.prune_ratio_train}")
    _logger.log_info(f"> prune_ratio_val:       {_params.prune_ratio_val}")
    _logger.log_info(f"> prune_ratio_test:      {_params.prune_ratio_test}")
    _logger.log_info("..")
    _logger.log_info(f"> pretrained_model_name: {_params.pretrained_model_name}")
    _logger.log_info(f"> uncased:               {_params.uncased}")
    _logger.log_info("..")
    _logger.log_info(f"> checkpoints:           {_params.checkpoints}")
    _logger.log_info(f"> logging_level:         {_params.logging_level}")
    _logger.log_info(f"> multiple_runs:         {_params.multiple_runs}")
    _logger.log_info(f"> seed:                  {_seed}")
    _logger.log_info("")
    _logger.log_info("- HPARAMS ----------------------------------------")
    _logger.log_info(f"> batch_size:       {_hparams.batch_size}")
    _logger.log_info(f"> max_seq_length:   {_hparams.max_seq_length}")
    _logger.log_info(f"> max_epochs:       {_hparams.max_epochs}")
    _logger.log_info(f"> early_stopping:   {_hparams.early_stopping}")
    _logger.log_info(f"> monitor:          {_hparams.monitor}")
    _logger.log_info(f"> min_delta:        {_hparams.min_delta}")
    _logger.log_info(f"> patience:         {_hparams.patience}")
    _logger.log_info(f"> mode:             {_hparams.mode}")
    _logger.log_info(f"> lr_max:           {_hparams.lr_max}")
    _logger.log_info(f"> lr_warmup_epochs: {_hparams.lr_warmup_epochs}")
    _logger.log_info(f"> lr_schedule:      {_hparams.lr_schedule}")
    _logger.log_info(f"> lr_num_cycles:    {_hparams.lr_num_cycles}")
    _logger.log_info("")


def _get_model_checkpoint_directory(_params):
    """
    :param _params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :return: model_checkpoint_directory [str]
    """
    return join(env_variable("DIR_CHECKPOINTS"), _params.experiment_run_name_nr)


def get_callbacks(_params, _hparams, _log_dirs):
    """
    :param _params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param _log_dirs:   [argparse.Namespace] attr: mlflow, tensorboard
    :return: _callbacks: [list] w/ [pytorch lightning callback]
    """
    early_stopping = vars(_hparams)["early_stopping"]

    if early_stopping:
        early_stopping_params = {
            k: vars(_hparams)[k] for k in ["monitor", "min_delta", "patience", "mode"]
        }
        _callbacks = [
            ModelCheckpoint(
                dirpath=_get_model_checkpoint_directory(_params),
                filename="{epoch}",
                monitor=vars(_hparams)["monitor"],
                verbose=True,
            ),
            EarlyStopping(**early_stopping_params, verbose=True),
        ]
    else:
        _callbacks = [
            ModelCheckpoint(
                dirpath=_get_model_checkpoint_directory(_params),
                filename="{epoch}",
                monitor=None,
                verbose=True,
                save_top_k=0,
                save_last=True,
            ),
        ]
        _callbacks[0].CHECKPOINT_NAME_LAST = "{epoch}"

    return _callbacks


def get_callback_info(_callbacks, _params, _hparams):
    """
    :param _callbacks: [dict] w/ keys 'checkpoint', 'early_stop' & values = [pytorch lightning callback]
    :param _params:    [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _hparams:   [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :return: _callback_info: [dict] w/ keys 'epoch_best', 'epoch_stopped', 'checkpoint_best'
    """
    callback_info = dict()

    early_stopping = hasattr(_callbacks[1], 'stopped_epoch')
    if early_stopping:
        checkpoint_best = list(_callbacks[0].best_k_models.keys())[0]
    else:
        checkpoint_best = _callbacks[0].last_model_path

    if early_stopping and _callbacks[1].stopped_epoch:
        callback_info["epoch_stopped"] = (
            _callbacks[1].stopped_epoch
        )
    else:
        callback_info["epoch_stopped"] = _hparams.max_epochs - 1

    callback_info["epoch_best"] = checkpoint2epoch(checkpoint_best)
    callback_info["checkpoint_best"] = join(
        _get_model_checkpoint_directory(_params),
        epoch2checkpoint(callback_info["epoch_best"]),
    )

    return callback_info


def logging_start(_params, _log_dirs):
    """
    :param _params:      [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
    :param _log_dirs:    [argparse.Namespace] attr: mlflow, tensorboard
    :return: _tb_logger: [pytorch lightning TensorBoardLogger]
    """
    # mlflow.tracking.set_tracking_uri(_log_dirs.mlflow)                # mlflow
    # mlflow.set_experiment(_params.experiment_name)                    # mlflow
    _tb_logger = TensorBoardLogger(
        save_dir=_log_dirs.tensorboard,  # tensorboard
        name=_params.experiment_run_name_nr,
    )
    return _tb_logger


def logging_end(
    _tb_logger, _callback_info, _hparams, _model_stopped, _model_best, _logger
):
    """
    :param _tb_logger:     [pytorch lightning TensorBoardLogger]
    :param _callback_info: [dict] w/ keys 'epoch_best', 'epoch_stopped', 'checkpoint_best'
    :param _hparams:       [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, lr_*
    :param _model_stopped: [NerModelTrain] epoch_stopped
    :param _model_best:    [NerModelTrain] epoch_best
    :param _logger:        [DefaultLogger]
    :return: -
    """
    epoch_best = _callback_info["epoch_best"]
    epoch_stopped = _callback_info["epoch_stopped"]

    _logger.log_info(f"epoch_best: {epoch_best}")
    _logger.log_info(f"epoch_stopped: {epoch_stopped}")

    _model_stopped.mlflow_client.log_metric("EPOCH_BEST", epoch_best)
    _model_stopped.mlflow_client.log_metric("EPOCH_STOPPED", epoch_stopped)

    for metric in ("token_fil_f1_micro", "entity_fil_f1_micro"):
        _model_stopped.mlflow_client.log_metric(
            f"epoch_best_val_{metric}".upper(),
            _model_stopped.epoch_metrics["val"][epoch_best][metric],
        )
        _model_stopped.mlflow_client.log_metric(
            f"epoch_best_test_{metric}".upper(),
            _model_best.epoch_metrics["test"][0][metric],
        )

    _model_stopped.mlflow_client.finish_artifact_logger()  # mlflow
    _tb_logger_stopped_epoch(
        _tb_logger, _hparams, epoch_best, epoch_stopped, _model_stopped, _model_best
    )  # tb


def _tb_logger_stopped_epoch(
    _tb_logger,
    _hparams,
    _epoch_best,
    _epoch_stopped,
    _model_stopped,
    _model_best,
    metrics=("token_fil_f1_micro", "entity_fil_f1_micro"),
):
    """
    log hparams and metrics for stopped epoch
    -----------------------------------------
    :param _tb_logger:      [pytorch lightning TensorboardLogger]
    :param _hparams:        [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
    :param _epoch_best:     [int]
    :param _epoch_stopped:  [int]
    :param _model_stopped:  [NerModelTrain] epoch_stopped
    :param _model_best:     [NerModelTrain] epoch_best
    :param metrics:         [tuple] of metrics to be logged in hparams section
    :return:
    """
    hparams_dict = dict()

    # epoch_best & epoch_stopped
    hparams_dict["hparam/train/epoch_best"] = _epoch_best
    hparams_dict["hparam/train/epoch_stopped"] = _epoch_stopped

    # val/test
    hparams_best_val = {
        f"hparam/val/epoch_best_{_epoch_best}/{metric}": _model_stopped.epoch_metrics[
            "val"
        ][_epoch_best][metric]
        for metric in metrics
    }

    hparams_best_test = {
        f"hparam/test/epoch_best_{_epoch_best}/{metric}": _model_best.epoch_metrics[
            "test"
        ][0][metric]
        for metric in metrics
    }

    hparams_dict.update(hparams_best_val)
    hparams_dict.update(hparams_best_test)

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


def remove_checkpoint(_checkpoint_path, _default_logger):
    """
    remove checkpoint stored at _checkpoint_path
    --------------------------------------------
    :param _checkpoint_path: [str], e.g. '[..]/data/results/checkpoints/exp_default/run1/epoch=0.ckpt'
    :param _default_logger:  [DefaultLogger]
    :return: -
    """
    os.remove(_checkpoint_path)
    _default_logger.log_info(f"> checkpoint {_checkpoint_path} removed")