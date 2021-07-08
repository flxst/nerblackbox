import warnings
import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report as classification_report_seqeval
from sklearn.metrics import classification_report as classification_report_sklearn
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn
from typing import List, Dict, Union, Tuple, Any, Optional

from nerblackbox.modules.ner_training.metrics.ner_metrics import NerMetrics
from nerblackbox.modules.ner_training.metrics.ner_metrics import (
    convert2bio,
    convert2plain,
)
from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import (
    order_tag_list,
    convert_tag_list_bio2plain,
)


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
            annotation_scheme: e.g. "plain", "bio"
            default_logger:
            logged_metrics:
        """
        self.current_epoch = current_epoch
        self.tag_list = order_tag_list(tag_list)
        self.tag_list_plain = convert_tag_list_bio2plain(tag_list)
        self.annotation_scheme = annotation_scheme
        self.default_logger = default_logger
        self.logged_metrics = logged_metrics

    ####################################################################################################################
    # 2. VALIDATE / COMPUTE METRICS
    ####################################################################################################################
    def execute(
        self, phase: str, outputs: List[Union[torch.Tensor, Dict[str, Any]]]
    ) -> Tuple[Dict[str, np.array], str, str, float]:
        """
        - validate on all batches of one epoch, i.e. whole val or test dataset

        Args:
            phase:   [str], 'val', 'test'
            outputs: [list] of [lists] = [batch_loss, batch_tag_ids, batch_logits] with 3 torch tensors for each batch

        Returns:
            epoch_metrics          [dict] w/ keys 'all_acc', 'fil_f1_micro', .. & values = [np array]
            classification_report: [str]
            confusion_matrix:      [str]
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
        if phase == "test":
            confusion_matrix = self._get_confusion_matrix_str(
                epoch_tags,
                phase=phase,
                epoch=self.current_epoch,
            )
            classification_report = self._get_classification_report(
                epoch_tags,
                phase=None,
                epoch=None,
            )
        else:
            classification_report = ""
            confusion_matrix = ""

        return epoch_metrics, classification_report, confusion_matrix, np_epoch["loss"]

    @staticmethod
    def _convert_output_to_np_batch(
        outputs: List[Union[torch.Tensor, Dict[str, Any]]],
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
        ] + self.tag_list_plain:
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
        special tags [*] have tag_id = -100 and are converted to [S]

        Args:
            _tag_ids: [np array] of shape [batch_size * seq_length] with [int] elements

        Returns:
            _tags:    [np array] of shape [batch_size * seq_length] with [str] elements
        """
        return np.array(
            [
                self.tag_list[int(tag_id)] if tag_id >= 0 else "[S]"
                for tag_id in _tag_ids
            ]
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
        _tags_plain = {
            field: convert2plain(
                _tags[field], convert_to_plain=self.annotation_scheme != "plain"
            )
            for field in ["true", "pred"]
        }

        tag_list, tag_list_indices = self._get_filtered_tags(tag_subset, _tags_plain)
        required_tag_groups = (
            [tag_subset] if tag_subset in ["all", "fil", "O"] else ["ind"]
        )
        required_phases = [_phase]

        if tag_subset == "O":
            levels = ["token"]
        else:
            levels = ["token", "entity"]

        _metrics = dict()
        for level in levels:
            required_levels = [level]
            metrics_to_compute = self.logged_metrics.get_metrics(
                required_tag_groups=required_tag_groups,
                required_phases=required_phases,
                required_levels=required_levels,
            )

            if len(metrics_to_compute):
                ner_metrics = NerMetrics(
                    _tags["true"] if level == "entity" else _tags_plain["true"],
                    _tags["pred"] if level == "entity" else _tags_plain["pred"],
                    tag_list=tag_list if level == "token" else None,
                    tag_index=tag_list_indices if level == "entity" else None,
                    level=level,
                    plain_tags=self.annotation_scheme == "plain"
                    if level == "entity"
                    else True,
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
                exclude=["numberofclasses", "loss"],
            ):
                _metrics[f"{level}_{tag_subset}_{metric_type}"] = results[metric_type]

            # micro
            for metric_type in self.logged_metrics.get_metrics(
                required_tag_groups=required_tag_groups,
                required_phases=required_phases,
                required_levels=required_levels,
                required_averaging_groups=["micro"],
                exclude=["numberofclasses"],
            ):
                if required_tag_groups in [["O"], ["ind"]]:
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
                required_averaging_groups=["macro"],
            ):
                _metrics[f"{level}_{tag_subset}_{metric_type}_macro"] = results[
                    f"{metric_type}_macro"
                ]

        return _metrics

    def _get_filtered_tags(
        self, _tag_subset: str, _tags_plain: Optional[Dict[str, np.array]] = None
    ) -> Tuple[List[str], Optional[int]]:
        """
        helper method for _compute_metrics()
        get list of filtered tags corresponding to _tag_subset name

        Args:
            _tag_subset: [str], e.g. 'all', 'fil', 'PER'
            _tags_plain: [dict] w/ keys 'true', 'pred'      & values = [np array]

        Returns:
            _filtered_tags:       list of filtered tags
            _filtered_tags_index: filtered tags index in case of single _filtered_tag, ignoring "O"
        """
        if _tag_subset == "all":
            _filtered_tags = self.tag_list_plain
            _filtered_tags_index = None
        elif _tag_subset == "fil":
            _filtered_tags = [tag for tag in self.tag_list_plain if tag != "O"]
            _filtered_tags_index = None
        else:
            assert _tags_plain is not None, f"ERROR! need to provide _tags_plain"
            tag_list_plain_filtered = [
                elem
                for elem in self.tag_list_plain
                if (elem in _tags_plain["true"] or elem in _tags_plain["pred"])
                and elem != "O"
            ]
            _filtered_tags = [_tag_subset]

            try:
                _filtered_tags_index_list = [tag_list_plain_filtered.index(_tag_subset)]
                assert len(_filtered_tags_index_list) == 1
                _filtered_tags_index = _filtered_tags_index_list[0]
            except ValueError:
                _filtered_tags_index = None

        return _filtered_tags, _filtered_tags_index

    ####################################################################################################################
    # 2. CLASSIFICATION REPORT #########################################################################################
    ####################################################################################################################
    def _get_classification_report(
        self,
        epoch_tags: Dict[str, np.array],
        phase: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> str:
        """
        - get token-based (sklearn) & chunk-based (seqeval) classification report

        Args:
            epoch_tags:     [dict] w/ keys 'true', 'pred'      & values = [np array]
            phase:          [str], 'train', 'val', 'test'
            epoch:          [int]

        Returns:
            classification_report: [str]
        """
        warnings.filterwarnings("ignore")

        epoch_tags_plain = {
            field: convert2plain(
                epoch_tags[field], convert_to_plain=self.annotation_scheme != "plain"
            )
            for field in ["true", "pred"]
        }

        # token-based classification report, plain tags
        tag_list_filtered, _ = self._get_filtered_tags("fil")
        classification_report: str = ""
        if phase is not None and epoch is not None:
            classification_report += f"\n>>> Phase: {phase} | Epoch: {epoch}"
        classification_report += (
            "\n--- token-based, plain tag (sklearn) classification report on fil ---\n"
        )
        classification_report += classification_report_sklearn(
            epoch_tags_plain["true"], epoch_tags_plain["pred"], labels=tag_list_filtered
        )

        # chunk-based classification report
        epoch_tags_chunk = dict()
        for field in ["true", "pred"]:
            epoch_tags_chunk[field] = convert2bio(
                epoch_tags[field], convert_to_bio=self.annotation_scheme == "plain"
            )
        self.default_logger.log_debug("> annotation_scheme:", self.annotation_scheme)
        self.default_logger.log_debug(
            "> epoch_tags_chunk[true]:", list(set(epoch_tags_chunk["true"]))
        )
        self.default_logger.log_debug(
            "> epoch_tags_chunk[pred]:", list(set(epoch_tags_chunk["pred"]))
        )

        classification_report += (
            "\n--- entity-based (seqeval) classification report on fil ---\n"
        )
        classification_report += classification_report_seqeval(
            [epoch_tags_chunk["true"]], [epoch_tags_chunk["pred"]], suffix=False
        )

        warnings.resetwarnings()

        return classification_report

    def _get_confusion_matrix_str(
        self,
        epoch_tags: Dict[str, np.array],
        phase: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> str:
        """
        - get token-based (sklearn) confusion matrix

        Args:
            epoch_tags:     [dict] w/ keys 'true', 'pred'      & values = [np array]
            phase:          [str], 'train', 'val', 'test'
            epoch:          [int]

        Returns:
            confusion_matrix_str:  [str] with confusion matrix as pd dataframe
        """
        warnings.filterwarnings("ignore")

        epoch_tags_plain = {
            field: convert2plain(
                epoch_tags[field], convert_to_plain=self.annotation_scheme != "plain"
            )
            for field in ["true", "pred"]
        }

        # token-based confusion matrix, plain tags
        confusion_matrix = confusion_matrix_sklearn(
            epoch_tags_plain["true"],
            epoch_tags_plain["pred"],
            labels=self.tag_list_plain,
        )

        df_confusion_matrix = pd.DataFrame(confusion_matrix)
        df_confusion_matrix.columns = self.tag_list_plain
        df_confusion_matrix.index = self.tag_list_plain

        confusion_matrix_str: str = ""
        if phase is not None and epoch is not None:
            confusion_matrix_str += f"\n>>> Phase: {phase} | Epoch: {epoch}\n"
        confusion_matrix_str += (
            "\n--- token-based, plain tag (sklearn) confusion matrix on all ---"
            + "\n... rows = ground truth | columns = predictions\n"
            + f"{df_confusion_matrix.to_string()}"
        )

        return confusion_matrix_str
