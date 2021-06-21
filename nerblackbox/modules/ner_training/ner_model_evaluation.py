import warnings
import numpy as np
import torch
from seqeval.metrics import classification_report as classification_report_seqeval
from sklearn.metrics import classification_report as classification_report_sklearn
from typing import List, Dict, Union, Tuple

from nerblackbox.modules.ner_training.metrics.ner_metrics import NerMetrics
from nerblackbox.modules.ner_training.metrics.ner_metrics import convert_to_chunk
from nerblackbox.modules.utils.env_variable import env_variable


class NerModelEvaluation:
    def __init__(
        self,
        current_epoch: int,
        tag_list: List[str],
        annotation_scheme: str,
        default_logger,
        logged_metrics,
    ):
        """
        Args:
            current_epoch:     e.g. 1
            tag_list:          e.g. ["O", "PER", "ORG"]
            annotation_scheme: e.g. "plain", "BIO"
            default_logger:
            logged_metrics:
        """
        self.current_epoch = current_epoch
        self.tag_list = tag_list
        self.annotation_scheme = annotation_scheme
        self.default_logger = default_logger
        self.logged_metrics = logged_metrics

    ####################################################################################################################
    # 2. VALIDATE / COMPUTE METRICS
    ####################################################################################################################
    def execute(
        self, phase: str, outputs: List[List[torch.Tensor]]
    ) -> Tuple[Dict[str, np.array], str, float]:
        """
        - validate on all batches of one epoch, i.e. whole val or test dataset

        Args:
            phase:   [str], 'val', 'test'
            outputs: [list] of [lists] = [batch_loss, batch_tag_ids, batch_logits] with 3 torch tensors for each batch

        Returns:
            epoch_metrics          [dict] w/ keys 'all_acc', 'fil_f1_micro', .. & values = [np array]
            classification_report: [str]
            epoch_loss:            [float] mean of of all batch losses
        """
        print()
        np_batch: Dict[str, List[np.array]] = self._convert_output_to_np_batch(outputs)
        np_epoch: Dict[
            str, Union[np.number, np.array]
        ] = self._combine_np_batch_to_np_epoch(np_batch)

        # epoch metrics
        epoch_metrics, epoch_tags = self._compute_metrics(phase, np_epoch)

        # classification report
        classification_report = self._get_classification_report(
            phase, self.current_epoch, epoch_tags
        )

        return epoch_metrics, classification_report, np_epoch["loss"]

    @staticmethod
    def _convert_output_to_np_batch(
        outputs: List[List[torch.Tensor]],
    ) -> Dict[str, List[np.array]]:
        """
        - converts pytorch lightning output to np_batch dictionary

        Args:
            outputs: [list] of [lists] = [batch_loss, batch_tag_ids, batch_logits] with 3 torch tensors for each batch

        Returns:
            np_batch: [dict] w/ key-value pairs:
                                'loss':     [list] of <batch_size> x [1D np array]s of length <seq_length>
                                'tag_ids':  [list] of <batch_size> x [1D np array]s of length <seq_length>
                                'logits'    [list] of <batch_size> x [2D np array]s of size   <seq_length x num_tags>
        """
        return {
            "loss": [output[0].detach().cpu().numpy() for output in outputs],
            "tag_ids": [
                output[1].detach().cpu().numpy() for output in outputs
            ],  # [batch_size, seq_length]
            "logits": [
                output[2].detach().cpu().numpy() for output in outputs
            ],  # [batch_size, seq_length, num_tags]
        }

    @staticmethod
    def _combine_np_batch_to_np_epoch(
        np_batch: Dict[str, List[np.array]]
    ) -> Dict[str, Union[np.number, np.array]]:
        """
        - combine np_batch to np_epoch

        Args:
            np_batch: [dict] w/ key-value pairs:
                                'loss':     [list] of <batch_size> x [1D np array]s of length  <seq_length>
                                'tag_ids':  [list] of <batch_size> x [1D np array]s of length  <seq_length>
                                'logits'    [list] of <batch_size> x [2D np array]s of size   [<seq_length>, <num_tags>]

        Returns:
            np_epoch: [dict] w/ key-value pairs:
                                'loss':     [np value]
                                'tag_ids':  [1D np array] of length      <batch_size> x <seq_length>
                                'logits'    [2D np array] of size shape [<batch_size> x <seq_length>, <num_tags>]
        """
        return {
            "loss": np.stack(np_batch["loss"]).mean(),
            "tag_ids": np.concatenate(
                np_batch["tag_ids"]
            ),  # shape: [dataset_size, seq_length]
            "logits": np.concatenate(
                np_batch["logits"]
            ),  # shape: [dataset_size, seq_length, num_tags]
        }

    ####################################################################################################################
    # 1. COMPUTE METRICS ###############################################################################################
    ####################################################################################################################
    def _compute_metrics(
        self, phase: str, _np_epoch: Dict[str, Union[np.number, np.array]]
    ) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
        """
        - compute loss, acc, f1 scores for size/phase = batch/train or epoch/val-test

        Args:
            phase:         [str], 'train', 'val', 'test'
            _np_epoch: [dict] w/ key-value pairs:
                                 'loss':     [np value]
                                 'tag_ids':  [1D np array] of length      <batch_size> x <seq_length>
                                 'logits':   [2D np array] of size shape [<batch_size> x <seq_length>, <num_tags>]

        Returns:
            _epoch_metrics  [dict] w/ keys 'all_acc', 'fil_f1_micro', .. & values = [np array]
            _epoch_tags     [dict] w/ keys 'true', 'pred'                & values = [np array]
        """
        # batch / dataset
        tag_ids = dict()
        tag_ids["true"], tag_ids["pred"] = self._reduce_and_flatten(
            _np_epoch["tag_ids"], _np_epoch["logits"]
        )

        tags = {
            field: self._convert_tag_ids_to_tags(tag_ids[field])
            for field in ["true", "pred"]
        }
        _epoch_tags = self._get_rid_of_special_tag_occurrences(tags)

        self.default_logger.log_debug("phase:", phase)
        self.default_logger.log_debug(
            "true:", np.shape(tags["true"]), list(set(tags["true"]))
        )
        self.default_logger.log_debug(
            "pred:", np.shape(tags["pred"]), list(set(tags["pred"]))
        )

        # batch / dataset metrics
        _epoch_metrics = {"token_all_loss": _np_epoch["loss"]}
        for tag_subset in [
            "all",
            "fil",
            "chk",
        ] + self.tag_list:
            _epoch_metrics.update(
                self._compute_metrics_for_tags_subset(
                    _epoch_tags, phase, tag_subset=tag_subset
                )
            )

        return _epoch_metrics, _epoch_tags

    @staticmethod
    def _reduce_and_flatten(
        _np_tag_ids: np.array, _np_logits: np.array
    ) -> Tuple[np.array, np.array]:
        """
        helper method for _compute_metrics()
        reduce _np_logits (3D -> 2D), flatten both np arrays (2D -> 1D)

        Args:
            _np_tag_ids: [np array] of shape [batch_size, seq_length]
            _np_logits:  [np array] of shape [batch_size, seq_length, num_tags]

        Returns:
            true_flat: [np array] of shape [batch_size * seq_length], _np_tag_ids               flattened
            pred_flat: [np array] of shape [batch_size * seq_length], _np_logits    reduced and flattened
        """
        true_flat = _np_tag_ids.flatten()
        pred_flat = np.argmax(_np_logits, axis=-1).flatten()
        return true_flat, pred_flat

    def _convert_tag_ids_to_tags(self, _tag_ids: np.array) -> np.array:
        """
        helper method for _compute_metrics()
        convert tag_ids (int) to tags (str)

        Args:
            _tag_ids: [np array] of shape [batch_size * seq_length] with [int] elements

        Returns:
            _tags:    [np array] of shape [batch_size * seq_length] with [str] elements
        """
        return np.array(
            [self.tag_list[int(tag_id)] if tag_id >= 0 else "[S]" for tag_id in _tag_ids]
        )

    @staticmethod
    def _get_rid_of_special_tag_occurrences(
        _tags: Dict[str, np.array]
    ) -> Dict[str, np.array]:
        """
        helper method for _compute_metrics()
        get rid of all elements where '[S]' occurs in true array

        Args:
            _tags:      [dict] w/ keys = 'true', 'pred' and
                                  values = [np array] of shape [batch_size * seq_length]

        Returns:
            _tags_new:  [dict] w/ keys = 'true', 'pred' and
                                  values = [np array] of shape [batch_size * seq_length - # of spec. token occurrences]
        """
        pad_indices = np.where(_tags["true"] == "[S]")
        return {key: np.delete(_tags[key], pad_indices) for key in ["true", "pred"]}

    def _compute_metrics_for_tags_subset(
        self, _tags: Dict[str, np.array], _phase: str, tag_subset: str
    ) -> Dict[str, float]:
        """
        helper method for _compute_metrics()
        compute metrics for tags subset (e.g. 'all', 'fil')

        Args:
            _tags:      [dict] w/ keys 'true', 'pred'      & values = [np array]
            _phase:     [str], 'train', 'val'
            tag_subset: [str], e.g. 'all', 'fil', 'PER'

        Returns:
            _metrics    [dict] w/ keys = metric (e.g. 'all_precision_micro') and value = [float]
        """
        tag_list = self._get_filtered_tags(tag_subset)
        required_tag_groups = [tag_subset] if tag_subset in ["all", "fil", "chk"] else ["ind"]
        required_phases = [_phase]
        levels = ["token", "entity"]
        # level = "chunk" if tag_subset == "chk" else "token"

        _metrics = dict()
        for level in levels:
            required_levels = [level]
            metrics_to_compute = self.logged_metrics.get_metrics(required_tag_groups=required_tag_groups,
                                                                 required_phases=required_phases,
                                                                 required_levels=required_levels)

            if len(metrics_to_compute):
                ner_metrics = NerMetrics(
                    _tags["true"],
                    _tags["pred"],
                    tag_list=tag_list,
                    level=level,
                    plain_tags=self.annotation_scheme == "plain",
                )
                ner_metrics.compute(metrics_to_compute)
                results = ner_metrics.results_as_dict()
            else:
                results = dict()

            # simple
            for metric_type in self.logged_metrics.get_metrics(
                required_tag_groups=required_tag_groups,
                required_phases=required_phases,
                required_levels=required_levels,
                required_averaging_groups=["simple"],
                exclude=["loss"],
            ):
                # if results[metric_type] is not None:
                _metrics[f"{level}_{tag_subset}_{metric_type}"] = results[metric_type]

            # micro
            for metric_type in self.logged_metrics.get_metrics(
                required_tag_groups=required_tag_groups,
                required_phases=required_phases,
                required_levels=required_levels,
                required_averaging_groups=["micro"]
            ):
                # if results[f'{metric_type}_micro'] is not None:
                if required_tag_groups == ["ind"]:
                    _metrics[f"{level}_{tag_subset}_{metric_type}"] = results[
                        f"{metric_type}_micro"
                    ]
                else:
                    _metrics[f"{level}_{tag_subset}_{metric_type}_micro"] = results[
                        f"{metric_type}_micro"
                    ]

            # macro
            for metric_type in self.logged_metrics.get_metrics(
                required_tag_groups=required_tag_groups,
                required_phases=required_phases,
                required_levels=required_levels,
                required_averaging_groups=["macro"]
            ):
                # if results[f'{metric_type}_macro'] is not None:
                _metrics[f"{level}_{tag_subset}_{metric_type}_macro"] = results[
                    f"{metric_type}_macro"
                ]

        return _metrics

    def _get_filtered_tags(self, _tag_subset: str) -> List[str]:
        """
        helper method for _compute_metrics()
        get list of filtered tags corresponding to _tag_subset name

        Args:
            _tag_subset: [str], e.g. 'all', 'fil', 'PER'

        Returns:
            _filtered_tags: [list] of filtered tags [str]
        """
        if _tag_subset == "all":
            return self.tag_list
        elif _tag_subset in ["fil", "chk"]:
            return [tag for tag in self.tag_list if tag != "O"]
        else:
            return [_tag_subset]

    ####################################################################################################################
    # 2. CLASSIFICATION REPORT #########################################################################################
    ####################################################################################################################
    def _get_classification_report(
        self, phase: str, epoch: int, epoch_tags: Dict[str, np.array]
    ) -> str:
        """
        - get token-based (sklearn) & chunk-based (seqeval) classification report

        Args:
            phase:          [str], 'train', 'val', 'test'
            epoch:          [int]
            epoch_tags:     [dict] w/ keys 'true', 'pred'      & values = [np array]

        Returns:
            classification_report: [str]
        """
        warnings.filterwarnings("ignore")

        # token-based classification report
        tag_list_filtered = self._get_filtered_tags("fil")
        classification_report: str = ""
        classification_report += f"\n>>> Phase: {phase} | Epoch: {epoch}"
        classification_report += (
            "\n--- token-based (sklearn) classification report on fil ---\n"
        )
        classification_report += classification_report_sklearn(
            epoch_tags["true"], epoch_tags["pred"], labels=tag_list_filtered
        )

        # chunk-based classification report
        epoch_tags_chunk = dict()
        for field in ["true", "pred"]:
            epoch_tags_chunk[field] = convert_to_chunk(
                epoch_tags[field], to_bio=self.annotation_scheme == "plain"
            )
        self.default_logger.log_debug("> annotation_scheme:", self.annotation_scheme)
        self.default_logger.log_debug(
            "> epoch_tags_chunk[true]:", list(set(epoch_tags_chunk["true"]))
        )
        self.default_logger.log_debug(
            "> epoch_tags_chunk[pred]:", list(set(epoch_tags_chunk["pred"]))
        )

        classification_report += (
            "\n--- chunk-based (seqeval) classification report on fil ---\n"
        )
        classification_report += classification_report_seqeval(
            epoch_tags_chunk["true"], epoch_tags_chunk["pred"], suffix=False
        )

        warnings.resetwarnings()

        return classification_report
