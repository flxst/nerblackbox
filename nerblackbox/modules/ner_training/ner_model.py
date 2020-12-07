import warnings
import numpy as np
from seqeval.metrics import classification_report as classification_report_seqeval
from sklearn.metrics import classification_report as classification_report_sklearn
import pytorch_lightning as pl
from abc import ABC, abstractmethod

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer

from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from nerblackbox.modules.ner_training.metrics.ner_metrics import NerMetrics
from nerblackbox.modules.ner_training.metrics.ner_metrics import convert_to_chunk
from nerblackbox.modules.utils.util_functions import split_parameters
from nerblackbox.modules.utils.env_variable import env_variable


class NerModel(pl.LightningModule, ABC):
    def __init__(self, hparams):
        """
        :param hparams: [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__()
        self.hparams = hparams

        # split up hparams
        self.params, self._hparams, self.log_dirs, self.experiment = split_parameters(
            hparams
        )

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
        pass

    def _preparations_data_general(self):
        """
        :created attr: tokenizer         [transformers AutoTokenizer]
        :created attr: data_preprocessor [DataPreprocessor]
        :return: -
        """
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, do_lower_case=False
        )  # needs to be False !!

        self.data_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            do_lower_case=self.params.uncased,  # can be True !!
            max_seq_length=self._hparams.max_seq_length,
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
            _input_ids,
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
        self.write_metrics_for_tensorboard("train", {"all+_loss": batch_train_loss})

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
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        # update params
        optimizer.step()
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

    def _create_scheduler(self, _lr_warmup_epochs, _lr_schedule, _lr_num_cycles=None):
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

        num_training_steps = self._get_steps(self._hparams.max_epochs)
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

    ####################################################################################################################
    # 2. VALIDATE / COMPUTE METRICS
    ####################################################################################################################
    def _validate_on_epoch(self, phase, outputs):
        """
        validate on all batches of one epoch, i.e. whole val or test dataset
        --------------------------------------------------------------------
        :param phase:   [str], 'val', 'test'
        :param outputs: [list] of [list] w/ 3 elements [batch_loss, batch_tag_ids, batch_logits] for each batch
        :return: [dict] w/ key '<phase>_loss' & value = mean batch loss [float]
        """
        # to cpu/numpy
        np_batch = {
            "loss": [output[0].detach().cpu().numpy() for output in outputs],
            "tag_ids": [
                output[1].detach().cpu().numpy() for output in outputs
            ],  # [batch_size, seq_length]
            "logits": [
                output[2].detach().cpu().numpy() for output in outputs
            ],  # [batch_size, seq_length, num_tags]
        }

        # combine np_batch metrics to np_epoch metrics
        np_epoch = {
            "loss": np.stack(np_batch["loss"]).mean(),
            "tag_ids": np.concatenate(
                np_batch["tag_ids"]
            ),  # shape: [epoch_size, seq_length]
            "logits": np.concatenate(
                np_batch["logits"]
            ),  # shape: [epoch_size, seq_length, num_tags]
        }

        # epoch metrics
        epoch_metrics, epoch_tags = self.compute_metrics(phase, np_epoch)

        # tracked metrics & classification reports
        self.add_epoch_metrics(
            phase, self.current_epoch, epoch_metrics
        )  # attr: epoch_metrics
        self.get_classification_report(
            phase, self.current_epoch, epoch_tags
        )  # attr: classification_reports

        # logging: tb
        self.write_metrics_for_tensorboard(phase, epoch_metrics)

        # logging: mlflow
        if phase == "val":
            self.mlflow_client.log_metrics(self.current_epoch, epoch_metrics)
        self.mlflow_client.log_classification_report(
            self.classification_reports[phase][self.current_epoch],
            overwrite=(phase == "val" and self.current_epoch == 0),
        )
        self.mlflow_client.finish_artifact_mlflow()

        # print
        self._print_metrics(
            phase, epoch_metrics, self.classification_reports[phase][self.current_epoch]
        )

        self.default_logger.log_debug(f"--> {phase}: epoch done")

        return {f"{phase}_loss": np_epoch["loss"]}

    def compute_metrics(self, phase, _np_dict):
        """
        computes loss, acc, f1 scores for size/phase = batch/train or epoch/val-test
        ----------------------------------------------------------------------------
        :param phase:          [str], 'train', 'val', 'test'
        :param _np_dict:       [dict] w/ key-value pairs:
                                     'loss':     [np value]
                                     'tag_ids':  [np array] of shape [batch_size, seq_length]
                                     'logits'    [np array] of shape [batch_size, seq_length, num_tags]
        :return: metrics       [dict] w/ keys 'all+_loss', 'all+_acc', 'fil_f1_micro', .. & values = [np array]
                 tags          [dict] w/ keys 'true', 'pred'      & values = [np array]
        """
        # batch / dataset
        tag_ids = dict()
        tag_ids["true"], tag_ids["pred"] = self._reduce_and_flatten(
            _np_dict["tag_ids"], _np_dict["logits"]
        )

        tags = {
            field: self._convert_tag_ids_to_tags(tag_ids[field])
            for field in ["true", "pred"]
        }
        tags = self._get_rid_of_pad_tag_occurrences(tags)

        self.default_logger.log_debug("phase:", phase)
        self.default_logger.log_debug(
            "true:", np.shape(tags["true"]), list(set(tags["true"]))
        )
        self.default_logger.log_debug(
            "pred:", np.shape(tags["pred"]), list(set(tags["pred"]))
        )

        if phase == "val":
            for field in ["true", "pred"]:
                np.save(f'{env_variable("DIR_RESULTS")}/{field}.npy', tags[field])

        # batch / dataset metrics
        metrics = {"all+_loss": _np_dict["loss"]}
        for tag_subset in [
            "all+",
            "all",
            "fil",
            "chk",
        ] + self.tag_list:  # self._get_filtered_tags():
            metrics.update(
                self._compute_metrics_for_tags_subset(
                    tags, phase, tag_subset=tag_subset
                )
            )

        return metrics, tags

    @staticmethod
    def _reduce_and_flatten(_np_tag_ids, _np_logits):
        """
        helper method
        reduce _np_logits (3D -> 2D), flatten both np arrays (2D -> 1D)
        ---------------------------------------------------------------
        :param _np_tag_ids: [np array] of shape [batch_size, seq_length]
        :param _np_logits:  [np array] of shape [batch_size, seq_length, num_tags]
        :return: true_flat: [np array] of shape [batch_size * seq_length], _np_tag_ids               flattened
                 pred_flat: [np array] of shape [batch_size * seq_length], _np_logits    reduced and flattened
        """
        true_flat = _np_tag_ids.flatten()
        pred_flat = np.argmax(_np_logits, axis=2).flatten()
        return true_flat, pred_flat

    def _convert_tag_ids_to_tags(self, _tag_ids):
        """
        helper method
        convert tag_ids (int) to tags (str)
        -----------------------------------
        :param _tag_ids: [np array] of shape [batch_size * seq_length] with [int] elements
        :return: _tags:  [np array] of shape [batch_size * seq_length] with [str] elements
        """
        return np.array([self.tag_list[elem] for elem in _tag_ids])

    @staticmethod
    def _get_rid_of_pad_tag_occurrences(_tags):
        """
        get rid of all elements where '[PAD]' occurs in true array
        ----------------------------------------------------------
        :param _tags:      [dict] w/ keys = 'true', 'pred' and
                                     values = [np array] of shape [batch_size * seq_length]
        :return: _tags_new [dict] w/ keys = 'true', 'pred' and
                                     values = [np array] of shape [batch_size * seq_length - # of pad occurrences]
        """
        pad_indices = np.where(_tags["true"] == "[PAD]")
        return {key: np.delete(_tags[key], pad_indices) for key in ["true", "pred"]}

    def _compute_metrics_for_tags_subset(self, _tags, _phase, tag_subset: str):
        """
        helper method
        compute metrics for tags subset (e.g. 'all', 'fil')
        ---------------------------------------------------
        :param _tags:      [dict] w/ keys 'true', 'pred'      & values = [np array]
        :param _phase:     [str], 'train', 'val'
        :param tag_subset: [str], e.g. 'all+', 'all', 'fil', 'PER'
        :return: _metrics  [dict] w/ keys = metric (e.g. 'all_precision_micro') and value = [float]
        """
        tag_list = self._get_filtered_tags(tag_subset)
        if tag_subset in ["all+", "all", "fil", "chk"]:
            tag_group = [tag_subset]
        else:
            tag_group = ["ind"]
        if tag_subset == "chk":
            level = "chunk"
        else:
            level = "token"

        ner_metrics = NerMetrics(
            _tags["true"],
            _tags["pred"],
            tag_list=tag_list,
            level=level,
            plain_tags=self.params.dataset_tags == "plain",
        )
        ner_metrics.compute(
            self.logged_metrics.get_metrics(tag_group=tag_group, phase_group=[_phase])
        )
        results = ner_metrics.results_as_dict()

        _metrics = dict()
        # simple
        for metric_type in self.logged_metrics.get_metrics(
            tag_group=tag_group,
            phase_group=[_phase],
            micro_macro_group=["simple"],
            exclude=["loss"],
        ):
            # if results[metric_type] is not None:
            _metrics[f"{tag_subset}_{metric_type}"] = results[metric_type]

        # micro
        for metric_type in self.logged_metrics.get_metrics(
            tag_group=tag_group, phase_group=[_phase], micro_macro_group=["micro"]
        ):
            # if results[f'{metric_type}_micro'] is not None:
            if tag_group == ["ind"]:
                _metrics[f"{tag_subset}_{metric_type}"] = results[
                    f"{metric_type}_micro"
                ]
            else:
                _metrics[f"{tag_subset}_{metric_type}_micro"] = results[
                    f"{metric_type}_micro"
                ]

        # macro
        for metric_type in self.logged_metrics.get_metrics(
            tag_group=tag_group, phase_group=[_phase], micro_macro_group=["macro"]
        ):
            # if results[f'{metric_type}_macro'] is not None:
            _metrics[f"{tag_subset}_{metric_type}_macro"] = results[
                f"{metric_type}_macro"
            ]

        return _metrics

    def _get_filtered_tags(self, _tag_subset):
        """
        helper method
        get list of filtered tags corresponding to _tag_subset name
        -----------------------------------------------------------
        :param _tag_subset: [str], e.g. 'all+', 'all', 'fil', 'PER'
        :return: _filtered_tags: [list] of filtered tags [str]
        """
        if _tag_subset == "all++":
            return None
        elif _tag_subset == "all+":
            return [tag for tag in self.tag_list if not tag == "[PAD]"]
        elif _tag_subset == "all":
            return [tag for tag in self.tag_list if not tag.startswith("[")]
        elif _tag_subset in ["fil", "chk"]:
            return [
                tag for tag in self.tag_list if not (tag.startswith("[") or tag == "O")
            ]
        else:
            return [_tag_subset]

    def add_epoch_metrics(self, phase, epoch, _epoch_metrics):
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

    def get_classification_report(self, phase, epoch, epoch_tags):
        """
        get token-based (sklearn) & chunk-based (seqeval) classification report
        -----------------------------------------------------------------------
        :param: epoch:                         [int]
        :param: epoch_tags:                    [dict] w/ keys 'true', 'pred'      & values = [np array]
        :changed attr: classification reports: [dict] w/ keys = epoch [int], values = classification report [str]
        :return: -
        """
        warnings.filterwarnings("ignore")

        self.classification_reports[phase][epoch] = ""

        # token-based classification report
        tag_list_filtered = self._get_filtered_tags("fil")
        self.classification_reports[phase][
            epoch
        ] += f"\n>>> Phase: {phase} | Epoch: {epoch}"
        self.classification_reports[phase][
            epoch
        ] += "\n--- token-based (sklearn) classification report on fil ---\n"
        self.classification_reports[phase][epoch] += classification_report_sklearn(
            epoch_tags["true"], epoch_tags["pred"], labels=tag_list_filtered
        )

        # chunk-based classification report
        epoch_tags_chunk = dict()
        for field in ["true", "pred"]:
            epoch_tags_chunk[field] = convert_to_chunk(
                epoch_tags[field], to_bio=self.params.dataset_tags == "plain"
            )
        self.default_logger.log_debug("> dataset_tags:", self.params.dataset_tags)
        self.default_logger.log_debug(
            "> epoch_tags_chunk[true]:", list(set(epoch_tags_chunk["true"]))
        )
        self.default_logger.log_debug(
            "> epoch_tags_chunk[pred]:", list(set(epoch_tags_chunk["pred"]))
        )

        self.classification_reports[phase][
            epoch
        ] += "\n--- chunk-based (seqeval) classification report on fil ---\n"
        self.classification_reports[phase][epoch] += classification_report_seqeval(
            epoch_tags_chunk["true"], epoch_tags_chunk["pred"], suffix=False
        )

        warnings.resetwarnings()

    def write_metrics_for_tensorboard(self, phase, metrics):
        """
        write metrics for tensorboard
        -----------------------------
        :param phase:         [str] 'train' or 'val'
        :param metrics:       [dict] w/ keys 'loss', 'acc', 'f1_macro_all', 'f1_micro_all'
        :return: -
        """
        # tb_logs: all
        tb_logs = {
            f'{phase}/{k.split("_", 1)[0].replace("+", "P")}/{k.split("_", 1)[1]}': v
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

    def _print_metrics(self, phase, _metrics, _classification_reports=None):
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
            "all+ loss:         {:.2f}".format(_metrics["all+_loss"])
        )
        self.default_logger.log_info(
            "all+ acc:          {:.2f}".format(_metrics["all+_acc"])
        )
        self.default_logger.log_debug(
            "all+ f1 (micro):   {:.2f}".format(_metrics["all+_f1_micro"])
        )
        self.default_logger.log_debug(
            "all+ f1 (macro):   {:.2f}".format(_metrics["all+_f1_macro"])
        )
        self.default_logger.log_info(
            "all  f1 (micro):   {:.2f}".format(_metrics["all_f1_micro"])
        )
        self.default_logger.log_debug(
            "all  f1 (macro):   {:.2f}".format(_metrics["all_f1_macro"])
        )
        self.default_logger.log_info(
            "fil  f1 (micro):   {:.2f}".format(_metrics["fil_f1_micro"])
        )
        self.default_logger.log_debug(
            "fil  f1 (macro):   {:.2f}".format(_metrics["fil_f1_macro"])
        )
        self.default_logger.log_info(
            "chk  f1 (micro):   {:.2f}".format(_metrics["chk_f1_micro"])
        )
        self.default_logger.log_info(f"-----------------------")
        if _classification_reports is not None:
            self.default_logger.log_debug(_classification_reports)
