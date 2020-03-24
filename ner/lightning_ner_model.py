
import os
import numpy as np
import pandas as pd
from seqeval.metrics import classification_report as classification_report_seqeval
from sklearn.metrics import classification_report as classification_report_sklearn
import pytorch_lightning as pl

from apex import amp
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForTokenClassification

from os.path import abspath, dirname
BASE_DIR = abspath(dirname(dirname(__file__)))

from ner.utils import util_functions
from ner.metrics.ner_metrics import NerMetrics
from ner.metrics.logged_metrics import LoggedMetrics
from ner.logging.mlflow_client import MLflowClient
from ner.logging.default_logger import DefaultLogger


class LightningNerModel(pl.LightningModule):

    def __init__(self,
                 params,
                 hparams,
                 log_dirs,
                 experiment=False):
        """
        :param params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        :param hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
        :param log_dirs:   [argparse.Namespace] attr: mlflow, tensorboard
        :param experiment: [bool] whether run is part of an experiment w/ multiple runs
        """
        super().__init__()
        self.params = params
        self._hparams = hparams
        self.log_dirs = log_dirs

        # logging
        self.default_logger = DefaultLogger(__file__, level=self.params.logging_level)  # python logging
        self.logged_metrics = LoggedMetrics()

        self.mlflow_client = MLflowClient(experiment_name=self.params.experiment_name,
                                          run_name=self.params.run_name,
                                          log_dir=self.log_dirs.mlflow,
                                          logged_metrics=self.logged_metrics.as_flat_list(),
                                          default_logger=self.default_logger)
        self.mlflow_client.log_params(params, hparams, experiment=experiment)

        self.epoch_valid_metrics = dict()
        self.classification_reports = dict()

        self._preparations()

        self.failures = None
        self.reset_failures()

    def _preparations(self):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.params.pretrained_model_name,
                                                  do_lower_case=False)  # needs to be False !!

        # data
        dataset_path = os.path.join(BASE_DIR, util_functions.get_dataset_path(self.params.dataset_name))
        self.dataloader, self.tag_list = util_functions.preprocess_data(dataset_path,
                                                                        tokenizer,
                                                                        batch_size=self._hparams.batch_size,
                                                                        do_lower_case=self.params.uncased,  # can be True !!
                                                                        max_seq_length=self._hparams.max_seq_length,
                                                                        prune_ratio=(self.params.prune_ratio_train,
                                                                                self.params.prune_ratio_valid),
                                                                        )
        # model
        self.model = AutoModelForTokenClassification.from_pretrained(self.params.pretrained_model_name,
                                                                     num_labels=len(self.tag_list))
        # optimizer
        self.optimizer = self._create_optimizer(self._hparams.lr_max,
                                                fp16=self.params.fp16)

        # learning rate
        self.scheduler = self._create_scheduler(self._hparams.lr_warmup_epochs,
                                                self._hparams.lr_schedule,
                                                self._hparams.lr_num_cycles)

    ####################################################################################################################
    # FORWARD & BACKWARD PROPAGATION
    ####################################################################################################################
    def forward(self, _input_ids, _input_mask, _segment_ids, _tag_ids):
        return self.model(_input_ids,
                          attention_mask=_input_mask,
                          token_type_ids=_segment_ids,
                          labels=_tag_ids,
                          )

    def backward(self, use_amp, loss, optimizer):
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    ####################################################################################################################
    # TRAIN
    ####################################################################################################################
    def training_step(self, batch, batch_idx):
        # REQUIRED
        batch = tuple(t.to(self.params.device) for t in batch)
        input_ids, input_mask, segment_ids, tag_ids = batch
        outputs = self.forward(input_ids, input_mask, segment_ids, tag_ids)

        batch_train_loss, logits = outputs[:2]

        # to cpu/numpy
        np_batch_train = {
            'loss': batch_train_loss.detach().cpu().numpy(),
            'tag_ids': tag_ids.to('cpu').numpy(),     # shape: [batch_size, seq_length]
            'logits': logits.detach().cpu().numpy(),  # shape: [batch_size, seq_length, num_tags]
        }

        # batch train metrics
        batch_train_metrics, _, batch_train_failures = self.compute_metrics('train', np_batch_train)

        # self.add_failures('batch', batch_train_failures)
        # self.print_failures('batch')

        # logging
        self.write_metrics_for_tensorboard('train', batch_train_metrics)

        return {'loss': batch_train_loss}

    def training_step_end(self):
        self.default_logger.log_debug('--> epoch training done')

    ####################################################################################################################
    # OPTIMIZER
    ####################################################################################################################
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return self.optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
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
        batch = tuple(t.to(self.params.device) for t in batch)
        input_ids, input_mask, segment_ids, tag_ids = batch
        outputs = self.forward(input_ids, input_mask, segment_ids, tag_ids)

        batch_valid_loss, logits = outputs[:2]

        # to cpu/numpy
        np_batch_valid = {
            'loss': batch_valid_loss.detach().cpu().numpy(),
            'tag_ids': tag_ids.to('cpu').numpy(),              # shape: [batch_size, seq_length]
            'logits': logits.detach().cpu().numpy(),           # shape: [batch_size, seq_length, num_tags]
        }

        return np_batch_valid

    def validation_end(self, outputs):
        # OPTIONAL

        # combine np_batch_valid metrics to np_epoch_valid metrics
        np_epoch_valid = {
            'loss': np.stack([np_batch_valid['loss'] for np_batch_valid in outputs]).mean(),
            'tag_ids': np.concatenate([np_batch_valid['tag_ids'] for np_batch_valid in outputs]),  # shape: [epoch_size, seq_length]
            'logits': np.concatenate([np_batch_valid['logits'] for np_batch_valid in outputs]),    # shape: [epoch_size, seq_length, num_tags]
        }
        # epoch metrics
        epoch_valid_metrics, epoch_valid_tag_ids, epoch_valid_failures = self.compute_metrics('valid', np_epoch_valid)

        self.add_failures('epoch', epoch_valid_failures)
        self.print_failures('epoch')
        self.reset_failures()

        # tracked metrics & classification reports
        self.add_epoch_valid_metrics(self.current_epoch, epoch_valid_metrics)    # attr: epoch_valid_metrics
        self.get_classification_report(self.current_epoch, epoch_valid_tag_ids)  # attr: classification_reports

        # logging
        self.write_metrics_for_tensorboard('valid', epoch_valid_metrics)                               # tb
        self.mlflow_client.log_metrics(self.current_epoch, epoch_valid_metrics)                        # mlflow
        self.mlflow_client.log_classification_report(self.classification_reports[self.current_epoch],
                                                     overwrite=self.current_epoch == 0)                # mlflow
        self.mlflow_client.finish_artifact()                                                           # mlflow

        # print
        self._print_metrics(epoch_valid_metrics, self.classification_reports[self.current_epoch])

        self.default_logger.log_debug('--> epoch validation done')

        return {'val_loss': np_epoch_valid['loss']}

    ####################################################################################################################
    # DATALOADER
    ####################################################################################################################
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.dataloader['train']

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.dataloader['valid']

    ####################################################################################################################
    # HELPER METHODS
    ####################################################################################################################
    def _create_optimizer(self, learning_rate, fp16=True, no_decay=('bias', 'gamma', 'beta')):
        """
        create optimizer with basic learning rate and L2 normalization for some parameters
        ----------------------------------------------------------------------------------
        :param learning_rate: [float] basic learning rate
        :param fp16:          [bool]
        :param no_decay:      [tuple of str] parameters that contain one of those are not subject to L2 normalization
        :return: optimizer:   [pytorch optimizer]
        """
        # Remove unused pooler that otherwise break Apex
        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.02},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # print('> param_optimizer')
        # print([n for n, p in param_optimizer])
        print('> {} parameters w/  weight decay'.format(len(optimizer_grouped_parameters[0]['params'])))
        print('> {} parameters w/o weight decay'.format(len(optimizer_grouped_parameters[1]['params'])))
        if fp16:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            # optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.learning_rate, bias_correction=False)
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            # optimizer = FusedAdam(optimizer_grouped_parameters, lr=learning_rate)

        # optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5, warmup=.1)
        return optimizer

    def _get_steps(self, _num_epochs):
        """
        gets steps = num_epochs * (number of training data samples)
        -----------------------------------------------------------------
        :param _num_epochs: [int], e.g. 10
        :return: steps:     [int], e.g. 2500 (in case of 250 training data samples)
        """
        return _num_epochs * len(self.dataloader['train'])

    def _create_scheduler(self, _lr_warmup_epochs, _lr_schedule, _lr_num_cycles=None):
        """
        create scheduler with warmup
        ----------------------------
        :param _lr_warmup_epochs:   [int]
        :param _lr_schedule:        [str], 'linear', 'constant', 'cosine', 'cosine_with_hard_resets'
        :param _lr_num_cycles:      [float, optional], e.g. 0.5, 1.0, only for cosine learning rate schedules
        :return: scheduler          [torch LambdaLR] learning rate scheduler
        """
        if _lr_schedule not in ['constant', 'linear', 'cosine', 'cosine_with_hard_restarts']:
            raise Exception(f'lr_schedule = {_lr_schedule} not implemented.')

        num_training_steps = self._get_steps(self._hparams.max_epochs)
        num_warmup_steps = self._get_steps(_lr_warmup_epochs)

        scheduler_params = {
            'num_warmup_steps': num_warmup_steps,
            'last_epoch': -1,
        }

        if _lr_schedule == 'constant':
            return get_constant_schedule_with_warmup(self.optimizer, **scheduler_params)
        else:
            scheduler_params['num_training_steps'] = num_training_steps

            if _lr_schedule == 'linear':
                return get_linear_schedule_with_warmup(self.optimizer, **scheduler_params)
            else:
                if _lr_num_cycles is not None:
                    scheduler_params['num_cycles'] = _lr_num_cycles  # else: use default values

                if _lr_schedule == 'cosine':
                    scheduler_params['num_training_steps'] = num_training_steps
                    return get_cosine_schedule_with_warmup(self.optimizer, **scheduler_params)
                elif _lr_schedule == 'cosine_with_hard_restarts':
                    scheduler_params['num_training_steps'] = num_training_steps
                    return get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, **scheduler_params)
                else:
                    raise Exception('create scheduler: logic is broken.')  # this should never happen

    ####################################################################################################################
    # 2. METRICS
    ####################################################################################################################
    def compute_metrics(self, phase, _np_dict):
        """
        computes loss, acc, f1 scores for size/phase = batch/train or epoch/valid
        -------------------------------------------------------------------------
        :param phase:          [str], 'train', 'valid'
        :param _np_dict:       [dict] w/ key-value pairs:
                                     'loss':     [np value]
                                     'tag_ids':  [np array] of shape [batch_size, seq_length]
                                     'logits'    [np array] of shape [batch_size, seq_length, num_tags]
        :return: metrics       [dict] w/ keys 'loss', 'acc', 'f1' & values = [np array]
                 tags_ids      [dict] w/ keys 'true', 'pred'      & values = [np array]
        """
        # batch / dataset
        tag_ids = dict()
        tag_ids['true'], tag_ids['pred'] = self._reduce_and_flatten(_np_dict['tag_ids'], _np_dict['logits'])
        self.default_logger.log_debug('phase:', phase)
        self.default_logger.log_debug('true:', np.shape(tag_ids['true']),
                                      [self.tag_list[i] for i in set(tag_ids['true'])])
        self.default_logger.log_debug('pred:', np.shape(tag_ids['pred']),
                                      [self.tag_list[i] for i in set(tag_ids['pred'])])

        # batch / dataset metrics
        metrics = {'all_loss': _np_dict['loss']}
        failures = dict()
        for tag_subset in ['all', 'fil'] + self._get_filtered_tags():
            _metrics, _failures = self._compute_metrics_for_tags_subset(tag_ids, phase, tag_subset=tag_subset)
            metrics.update(_metrics)

            failures[tag_subset] = _failures

        return metrics, tag_ids, failures

    def _get_filtered_tags(self):
        return [tag
                for tag in self.tag_list
                if not (tag.startswith('[') or tag == 'O')]

    def _get_filtered_tag_ids(self):
        return [self.tag_list.index(tag)
                for tag in self.tag_list
                if not (tag.startswith('[') or tag == 'O')]

    def _get_individual_tag_id(self, tag):
        return [self.tag_list.index(tag)]

    def _compute_metrics_for_tags_subset(self, _tag_ids, _phase, tag_subset: str):
        """
        helper method
        compute metrics for tags subset (e.g. 'all', 'fil')
        ---------------------------------------------------
        :param _tag_ids:   [dict] w/ keys 'true', 'pred'      & values = [np array]
        :param _phase:     [str], 'train', 'valid'
        :param tag_subset: [str], e.g. 'all', 'fil', 'PER'
        :return: _metrics  [dict] w/ keys = metric (e.g. 'all_precision_micro') and value = [float]
        """
        if tag_subset == 'all':
            tag_list = None
            tag_group = ['all']
        elif tag_subset == 'fil':
            tag_list = self._get_filtered_tag_ids()
            tag_group = ['fil']
        else:
            tag_list = self._get_individual_tag_id(tag_subset)
            tag_group = ['ind']

        ner_metrics = NerMetrics(_tag_ids['true'], _tag_ids['pred'], tag_list=tag_list)
        ner_metrics.compute(self.logged_metrics.get_metrics(tag_group=tag_group,
                                                            phase_group=[_phase]))
        results = ner_metrics.results_as_dict()
        failures = ner_metrics.failures_as_dict()

        _metrics = dict()
        # simple
        for metric_type in self.logged_metrics.get_metrics(tag_group=tag_group,
                                                           phase_group=[_phase],
                                                           micro_macro_group=['simple'],
                                                           exclude=['loss']):
            # if results[metric_type] is not None:
            _metrics[f'{tag_subset}_{metric_type}'] = results[metric_type]

        # micro
        for metric_type in self.logged_metrics.get_metrics(tag_group=tag_group,
                                                           phase_group=[_phase],
                                                           micro_macro_group=['micro']):
            # if results[f'{metric_type}_micro'] is not None:
            if tag_group == ['ind']:
                _metrics[f'{tag_subset}_{metric_type}'] = results[f'{metric_type}_micro']
            else:
                _metrics[f'{tag_subset}_{metric_type}_micro'] = results[f'{metric_type}_micro']

        # macro
        for metric_type in self.logged_metrics.get_metrics(tag_group=tag_group,
                                                           phase_group=[_phase],
                                                           micro_macro_group=['macro']):
            # if results[f'{metric_type}_macro'] is not None:
            _metrics[f'{tag_subset}_{metric_type}_macro'] = results[f'{metric_type}_macro']

        return _metrics, failures

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

    def add_epoch_valid_metrics(self, epoch, _epoch_valid_metrics):
        """
        add _epoch_valid_metrics to attribute/dict epoch_valid_metrics
        --------------------------------------------------------------
        :param: epoch:                      [int]
        :param: _epoch_valid_metrics:       [dict] w/ keys 'loss', 'acc', 'f1' & values = [np array]
        :changed attr: epoch_valid_metrics: [dict] w/ keys = epoch [int], values = _epoch_valid_metrics [dict]
        :return: -
        """
        self.epoch_valid_metrics[epoch] = _epoch_valid_metrics

    def get_classification_report(self, epoch, _epoch_valid_tag_ids):
        """
        get token-based (sklearn) & chunk-based (seqeval) classification report
        -----------------------------------------------------------------------
        :param: epoch:                   [int]
        :param: _epoch_valid_tag_ids:    [dict] w/ keys 'true', 'pred'      & values = [np array]
        :changed attr: classification reports: [dict] w/ keys = epoch [int], values = classification report [str]
        :return: -
        """
        import warnings
        warnings.filterwarnings("ignore")

        self.classification_reports[epoch] = ''

        # use tags instead of tag_ids
        epoch_valid_tags = {
            field: [self.tag_list[tag_id] for tag_id in _epoch_valid_tag_ids[field]]
            for field in ['true', 'pred']
        }

        # token-based classification report
        selected_tags = [tag for tag in self.tag_list if tag != 'O' and not tag.startswith('[')]
        self.classification_reports[epoch] += f'\n>>> Epoch: {epoch}'
        self.classification_reports[epoch] += '\n--- token-based (sklearn) classification report ---\n'
        self.classification_reports[epoch] += classification_report_sklearn(epoch_valid_tags['true'],
                                                                            epoch_valid_tags['pred'],
                                                                            labels=selected_tags)

        # enrich pred_tags & valid_tags with bio prefixes
        epoch_valid_tags_bio = {
            field: util_functions.add_bio_to_tag_list(util_functions.get_rid_of_special_tokens(epoch_valid_tags[field]))
            for field in ['true', 'pred']
        }

        # chunk-based classification report
        self.classification_reports[epoch] += '\n--- chunk-based (seqeval) classification report ---\n'
        self.classification_reports[epoch] += classification_report_seqeval(epoch_valid_tags_bio['true'],
                                                                            epoch_valid_tags_bio['pred'],
                                                                            suffix=False)

        warnings.resetwarnings()

    ####################################################################################################################
    # 2b. METRICS LOGGING
    ####################################################################################################################
    def _print_metrics(self, _metrics, _classification_reports=None):
        epoch_str = f'Epoch #{self.current_epoch} valid '
        self.default_logger.log_info('{} all loss:         {:.2f}'.format(epoch_str, _metrics['all_loss']))
        self.default_logger.log_debug('{} all acc:          {:.2f}'.format(epoch_str, _metrics['all_acc']))
        self.default_logger.log_debug('{} all f1 (macro):   {:.2f}'.format(epoch_str, _metrics['all_f1_macro']))
        self.default_logger.log_debug('{} all f1 (micro):   {:.2f}'.format(epoch_str, _metrics['all_f1_micro']))
        self.default_logger.log_debug('{} fil f1 (macro):   {:.2f}'.format(epoch_str, _metrics['fil_f1_macro']))
        self.default_logger.log_debug('{} fil f1 (micro):   {:.2f}'.format(epoch_str, _metrics['fil_f1_micro']))
        if _classification_reports is not None:
            self.default_logger.log_debug(_classification_reports)

    def write_metrics_for_tensorboard(self, phase, metrics):
        """
        write metrics for tensorboard
        -----------------------------
        :param phase:         [str] 'train' or 'valid'
        :param metrics:       [dict] w/ keys 'loss', 'acc', 'f1_macro_all', 'f1_micro_all'
        :return: -
        """
        # tb_logs: all
        tb_logs = {f'{phase}/{k}': v for k, v in metrics.items()}

        # tb_logs: learning rate
        if phase == 'train':
            tb_logs[f'{phase}/learning_rate'] = self.scheduler.get_last_lr()[0]

        # tb_logs
        self.logger.log_metrics(tb_logs, self.global_step)

    def add_failures(self, phase, failures):
        if self.failures['count'][phase] == 0:
            for tag_subset, _metric_failure_dict in failures.items():
                self.failures[phase][tag_subset] = _metric_failure_dict
        else:
            for tag_subset, _metric_failure_dict in failures.items():
                for k, v in self.failures[phase][tag_subset].items():
                    self.failures[phase][tag_subset][k] += v
        self.failures['count'][phase] += 1

    def print_failures(self, phase):
        columns = ['acc', 'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro', 'f1_micro', 'f1_macro']

        df = pd.DataFrame(columns=columns)
        self.default_logger.log_debug('-------------------------------')
        phase_keywork = 'TRAINING' if phase == 'batch' else 'VALIDATION'
        self.default_logger.log_debug(f'FAILURES FOR {phase_keywork}:')
        self.default_logger.log_debug('count =', self.failures['count'][phase])
        for tag_subset, _metric_failure_dict in self.failures[phase].items():
            # print(f'--> tag_subset = {tag_subset}')
            for k, v in _metric_failure_dict.items():
                # print(f'{k}: {v}')
                df.loc[tag_subset, k] = v
        for column in columns:
            df[column] = df[column].astype(int)
        self.default_logger.log_debug(df)

    def reset_failures(self):
        self.failures = {'batch': dict(),
                         'epoch': dict(),
                         'count': {'batch': 0, 'epoch': 0}}
