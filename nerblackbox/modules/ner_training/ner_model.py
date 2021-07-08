import os
from os.path import join
import numpy as np
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Optional, Callable, Union, Any
from omegaconf import OmegaConf

from torch.optim.lr_scheduler import LambdaLR

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer

from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from nerblackbox.modules.utils.util_functions import split_parameters
from nerblackbox.modules.ner_training.ner_model_evaluation import NerModelEvaluation
from nerblackbox.modules.ner_training.logging.mlflow_client import MLflowClient
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger

NEWLINE_TOKENS = ["[newline]", "[NEWLINE]"]


class NerModel(pl.LightningModule, ABC):
    def __init__(self, hparams: OmegaConf):
        """
        :param hparams: attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        # split up hparams
        (
            self.params,
            self.hyperparameters,
            self.log_dirs,
            self.experiment,
        ) = split_parameters(hparams)

        # preparations
        self._preparations()

    ####################################################################################################################
    # Abstract Base Methods ############################################################################################
    ####################################################################################################################
    @abstractmethod
    def _preparations(self):
        """
        :created attr: default_logger    [DefaultLogger]
        :created attr: logged_metrics    [LoggedMetrics]
        :created attr: tokenizer         [transformers AutoTokenizer]
        :created attr: data_preprocessor [DataPreprocessor]
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]

        [only train]
        :created attr: mlflow_client          [MLflowClient]
        :created attr: epoch_metrics          [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: classification_reports [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: pretrained_model_name  [str]
        :created attr: dataloader             [dict] w/ keys 'train', 'val', 'test' & values = [torch Dataloader]
        :created attr: optimizer              [torch optimizer]
        :created attr: scheduler              [torch LambdaLR]
        :return: -
        """
        self.mlflow_client: MLflowClient
        self.default_logger: DefaultLogger
        self.scheduler: LambdaLR
        self.tag_list: List[str]

    def _preparations_data_general(self):
        """
        :created attr: pretrained_model_name  [str]
        :created attr: tokenizer              [transformers AutoTokenizer]
        :created attr: data_preprocessor      [DataPreprocessor]
        :return: -
        """
        # 1. pretrained model name
        try:
            # use transformers model
            AutoTokenizer.from_pretrained(
                self.params.pretrained_model_name, use_fast=False
            )
            self.pretrained_model_name = self.params.pretrained_model_name
        except ValueError:
            # use local model
            self.pretrained_model_name = join(
                os.environ.get("DATA_DIR"),
                "pretrained_models",
                self.params.pretrained_model_name,
            )

        # 2. tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            do_lower_case=False,
            additional_special_tokens=NEWLINE_TOKENS,
            use_fast=True,
        )  # do_lower_case needs to be False !!

        # 3. data_preprocessor
        self.data_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            do_lower_case=self.params.uncased,  # can be True !!
            max_seq_length=self.hyperparameters.max_seq_length,
            default_logger=self.default_logger,
        )

    ####################################################################################################################
    # TRAIN/VAL/TEST
    ####################################################################################################################
    ####################################################################################################################
    # FORWARD & BACKWARD PROPAGATION
    ####################################################################################################################
    def forward(self, _input_ids, _attention_mask, _segment_ids, _tag_ids):
        """
        :param _input_ids:            [torch tensor] of shape [batch_size, seq_length],
                                             e.g. 1st row = [1, 567, 568, 569, .., 2, 611, 612, .., 2, 0, 0, 0, ..]
        :param _attention_mask:       [torch tensor] of shape [batch_size, seq_length],
                                             e.g. 1st row = [1,   1,   1,   1, .., 1,   1,   1, .., 1, 0, 0, 0, ..]
        :param _segment_ids:          [torch tensor] of shape [batch_size, seq_length],
                                             e.g. 1st row = [0,   0,   0,   0, .., 0,   1,   1, .., 1, 0, 0, 0, ..]
        :param _tag_ids:              [torch tensor] of shape [batch_size, seq_length],
                                             e.g. 1st row = [1,   3,   3,   4, .., 2,   3,   3, .., 2, 0, 0, 0, ..]
        :return: _outputs: [list] of 2 elements:
                    i)  _loss: [float] cross entropy between _tag_ids & _tags_ids_predictions
                                       on non-padding tokens (i.e. where elements in _input_ids are not 0)
                    ii) _tag_ids_prediction_logits: [torch tensor] of shape [batch_size, seq_length, vocabulary_size]
        """
        return self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            token_type_ids=_segment_ids,
            labels=_tag_ids,
        )

    ####################################################################################################################
    # TRAIN
    ####################################################################################################################
    def training_step(self, batch, batch_idx):
        # REQUIRED
        input_ids, attention_mask, segment_ids, tag_ids = batch
        outputs = self(input_ids, attention_mask, segment_ids, tag_ids)
        batch_train_loss = outputs[0]

        # logging
        self._write_metrics_for_tensorboard("train", {"all_loss": batch_train_loss})

        # debug
        if batch_idx == 0:
            self._debug_step_check(
                "train", batch, outputs, input_ids, attention_mask, segment_ids, tag_ids
            )

        return {"loss": batch_train_loss}

    ####################################################################################################################
    # OPTIMIZER
    ####################################################################################################################
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return self.optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:

        # update params
        if optimizer is not None:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        # update learning rate
        self.scheduler.step()

    ####################################################################################################################
    # VALID
    ####################################################################################################################
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        input_ids, attention_mask, segment_ids, tag_ids = batch
        outputs = self(input_ids, attention_mask, segment_ids, tag_ids)
        batch_loss, logits = outputs[:2]

        # debug
        if batch_idx == 0:
            self._debug_step_check(
                "val", batch, outputs, input_ids, attention_mask, segment_ids, tag_ids
            )

        return batch_loss, tag_ids, logits

    def validation_epoch_end(self, outputs):
        """
        :param outputs: [list] of [list] w/ 3 elements [batch_loss, batch_tag_ids, batch_logits] for each batch
        :return:        [dict] w/ key 'val_loss' & value = mean batch loss of val dataset [float]
        """
        # OPTIONAL
        return self._validate_on_epoch("val", outputs=outputs)

    ####################################################################################################################
    # TEST
    ####################################################################################################################
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        input_ids, attention_mask, segment_ids, tag_ids = batch
        outputs = self(input_ids, attention_mask, segment_ids, tag_ids)
        batch_loss, logits = outputs[:2]

        # debug
        if batch_idx == 0:
            self._debug_step_check(
                "test", batch, outputs, input_ids, attention_mask, segment_ids, tag_ids
            )

        return batch_loss, tag_ids, logits

    def test_epoch_end(self, outputs):
        """
        :param outputs: [list] of [list] w/ 3 elements [batch_loss, batch_tag_ids, batch_logits] for each batch
        :return:        [dict] w/ key 'test_loss' & value = mean batch loss of test dataset [float]
        """
        # OPTIONAL
        return self._validate_on_epoch("test", outputs=outputs)

    ####################################################################################################################
    # DATALOADER
    ####################################################################################################################
    def train_dataloader(self):
        # REQUIRED
        return self.dataloader["train"]

    def val_dataloader(self):
        # OPTIONAL
        return self.dataloader["val"]

    def test_dataloader(self):
        # OPTIONAL
        return self.dataloader["test"]

    ####################################################################################################################
    # HELPER METHODS
    # 1. PREPARATIONS
    # 2. VALIDATE / COMPUTE METRICS
    # 3. PRINT / LOG
    ####################################################################################################################

    ####################################################################################################################
    # 1. PREPARATIONS
    ####################################################################################################################
    def _create_optimizer(
        self, learning_rate, fp16=True, no_decay=("bias", "gamma", "beta")
    ):
        """
        create optimizer with basic learning rate and L2 normalization for some parameters
        ----------------------------------------------------------------------------------
        :param learning_rate: [float] basic learning rate
        :param fp16:          [bool]
        :param no_decay:      [tuple of str] parameters that contain one of those are not subject to L2 normalization
        :return: optimizer:   [torch optimizer]
        """
        # Remove unused pooler that otherwise break Apex
        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.02,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # print('> param_optimizer')
        # print([n for n, p in param_optimizer])
        self.default_logger.log_debug(
            "> parameters w/  weight decay:",
            len(optimizer_grouped_parameters[0]["params"]),
        )
        self.default_logger.log_debug(
            "> parameters w/o weight decay:",
            len(optimizer_grouped_parameters[1]["params"]),
        )
        if fp16:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            # optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.learning_rate, bias_correction=False)
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            # optimizer = FusedAdam(optimizer_grouped_parameters, lr=learning_rate)

        # optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5, warmup=.1)
        return optimizer

    def _create_scheduler(
        self, _lr_warmup_epochs, _lr_schedule, _lr_num_cycles=None
    ) -> LambdaLR:
        """
        create scheduler with warmup
        ----------------------------
        :param _lr_warmup_epochs:   [int]
        :param _lr_schedule:        [str], 'linear', 'constant', 'cosine', 'cosine_with_hard_resets'
        :param _lr_num_cycles:      [float, optional], e.g. 0.5, 1.0, only for cosine learning rate schedules
        :return: scheduler          [torch LambdaLR] learning rate scheduler
        """
        if _lr_schedule not in [
            "constant",
            "linear",
            "cosine",
            "cosine_with_hard_restarts",
        ]:
            raise Exception(f"lr_schedule = {_lr_schedule} not implemented.")

        num_training_steps = self._get_steps(self.hyperparameters.max_epochs)
        num_warmup_steps = self._get_steps(_lr_warmup_epochs)

        scheduler_params = {
            "num_warmup_steps": num_warmup_steps,
            "last_epoch": -1,
        }

        if _lr_schedule == "constant":
            return get_constant_schedule_with_warmup(self.optimizer, **scheduler_params)
        else:
            scheduler_params["num_training_steps"] = num_training_steps

            if _lr_schedule == "linear":
                return get_linear_schedule_with_warmup(
                    self.optimizer, **scheduler_params
                )
            else:
                if _lr_num_cycles is not None:
                    scheduler_params[
                        "num_cycles"
                    ] = _lr_num_cycles  # else: use default values

                if _lr_schedule == "cosine":
                    scheduler_params["num_training_steps"] = num_training_steps
                    return get_cosine_schedule_with_warmup(
                        self.optimizer, **scheduler_params
                    )
                elif _lr_schedule == "cosine_with_hard_restarts":
                    scheduler_params["num_training_steps"] = num_training_steps
                    return get_cosine_with_hard_restarts_schedule_with_warmup(
                        self.optimizer, **scheduler_params
                    )
                else:
                    raise Exception(
                        "create scheduler: logic is broken."
                    )  # this should never happen

    def _get_steps(self, _num_epochs):
        """
        helper method for _create_scheduler
        gets steps = num_epochs * (number of training data samples)
        -----------------------------------------------------------------
        :param _num_epochs: [int], e.g. 10
        :return: steps:     [int], e.g. 2500 (in case of 250 training data samples)
        """
        return _num_epochs * len(self.dataloader["train"])

    def _validate_on_epoch(
        self, phase: str, outputs: List[Union[torch.Tensor, Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Args:
            phase:   [str], 'val', 'test'
            outputs: [list] of [lists] = [batch_loss, batch_tag_ids, batch_logits] with 3 torch tensors for each batch

        Returns:
            [dict] w/ key '<phase>_loss' & value = mean batch loss [float]
        """
        ner_model_evaluation = NerModelEvaluation(
            current_epoch=self.current_epoch,
            tag_list=self.tag_list,
            annotation_scheme=self.params.dataset_tags
            if self.params.dataset_tags != "iob"
            else "bio",  # TODO: clean
            default_logger=self.default_logger,
            logged_metrics=self.logged_metrics,
        )
        (
            epoch_metrics,
            classification_report,
            confusion_matrix,
            epoch_loss,
        ) = ner_model_evaluation.execute(phase, outputs)
        self._log_metrics_confusion_matrix_classification_report(
            phase,
            epoch_metrics,
            confusion_matrix,
            classification_report,
        )
        self.log(f"{phase}_loss", epoch_loss)  # for early stopping callback

        return {f"{phase}_loss": epoch_loss}

    def _log_metrics_confusion_matrix_classification_report(
        self,
        phase: str,
        epoch_metrics: Dict[str, np.array],
        confusion_matrix: Optional[str] = "",
        classification_report: Optional[str] = "",
    ) -> None:
        """
        Args:
            phase:                  'val', 'test'
            epoch_metrics           keys 'all_acc', 'fil_f1_micro', .. & values = [np array]
            confusion_matrix:
            classification_report:
        """
        # tracked metrics & classification reports
        self._add_epoch_metrics(
            phase, self.current_epoch, epoch_metrics
        )  # attr: epoch_metrics

        # logging: tb
        self._write_metrics_for_tensorboard(phase, epoch_metrics)

        # logging: mlflow
        if phase == "test":
            self.mlflow_client.log_metrics(self.current_epoch, epoch_metrics)

            if confusion_matrix is not None:
                self.mlflow_client.log_artifact(
                    confusion_matrix,
                    overwrite=True,
                )
            if classification_report is not None:
                self.mlflow_client.log_artifact(
                    classification_report,
                    overwrite=False,
                )

            self.mlflow_client.finish_artifact_mlflow()

        # print
        self._print_metrics(phase, epoch_metrics, classification_report)

        self.default_logger.log_debug(f"--> {phase}: epoch done")

    def _add_epoch_metrics(self, phase, epoch, _epoch_metrics):
        """
        add _epoch_metrics to attribute/dict epoch_<phase>_metrics
        --------------------------------------------------------------
        :param: epoch:                [int]
        :param: _epoch_metrics:       [dict] w/ keys 'loss', 'acc', 'f1' & values = [np array]
        :changed attr: epoch_metrics: [dict] w/ keys = 'val'/'test and
                                                values = dict w/ keys = epoch [int], values = _epoch_metrics [dict]
        :return: -
        """
        self.epoch_metrics[phase][epoch] = _epoch_metrics

    def _write_metrics_for_tensorboard(self, phase, metrics):
        """
        write metrics for tensorboard
        -----------------------------
        :param phase:         [str] 'train' or 'val'
        :param metrics:       [dict] w/ keys 'loss', 'acc', 'f1_macro_all', 'f1_micro_all'
        :return: -
        """
        # tb_logs: all
        tb_logs = {
            f'{phase}/{k.split("_", 1)[0]}/{k.split("_", 1)[1]}': v
            for k, v in metrics.items()
        }

        # tb_logs: learning rate
        if phase == "train":
            tb_logs[f"{phase}/learning_rate"] = self.scheduler.get_last_lr()[0]

        # tb_logs
        self.logger.log_metrics(tb_logs, self.global_step)

    ####################################################################################################################
    # 3. PRINT / LOG
    ####################################################################################################################
    def _debug_step_check(
        self,
        phase,
        _batch,
        _outputs,
        _input_ids,
        _attention_mask,
        _segment_ids,
        _tag_ids,
    ):
        self.default_logger.log_debug(f"{phase.upper()} STEP CHECK")
        self.default_logger.log_debug(f"batch on gpu:   {_batch[0].is_cuda}")
        self.default_logger.log_debug(f"outputs on gpu: {_outputs[0].is_cuda}")
        self.default_logger.log_debug(
            f"input_ids      shape|1st row: {_input_ids.shape} | {_input_ids[0]}"
        )
        self.default_logger.log_debug(
            f"attention_mask shape|1st row: {_attention_mask.shape} | {_attention_mask[0]}"
        )
        self.default_logger.log_debug(
            f"segment_ids    shape|1st row: {_segment_ids.shape} | {_segment_ids[0]}"
        )
        self.default_logger.log_debug(
            f"tag_ids        shape|1st row: {_tag_ids.shape} | {_tag_ids[0]}"
        )

    def _print_metrics(self, phase, _metrics, _classification_reports: Optional[str]):
        """
        :param phase:         [str] 'train' or 'val'
        :param _metrics:
        :param _classification_reports:
        :return:
        """
        self.default_logger.log_info("")
        self.default_logger.log_info(
            f"--- Epoch #{self.current_epoch} {phase.ljust(5).upper()} ----"
        )
        self.default_logger.log_info(
            "token  all loss:         {:.2f}".format(_metrics["token_all_loss"])
        )
        self.default_logger.log_info(
            "token  all acc:          {:.2f}".format(_metrics["token_all_acc"])
        )
        self.default_logger.log_info(
            "token  all f1 (micro):   {:.2f}".format(_metrics["token_all_f1_micro"])
        )
        self.default_logger.log_info(
            "token  fil f1 (micro):   {:.2f}".format(_metrics["token_fil_f1_micro"])
        )
        self.default_logger.log_info(
            "entity fil f1 (micro):   {:.2f}".format(_metrics["entity_fil_f1_micro"])
        )
        self.default_logger.log_info(f"-----------------------")
        if _classification_reports is not None:
            self.default_logger.log_debug(_classification_reports)
