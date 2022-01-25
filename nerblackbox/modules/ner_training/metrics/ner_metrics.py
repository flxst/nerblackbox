from dataclasses import dataclass
from dataclasses import asdict
from typing import List, Tuple, Callable

import numpy as np
from sklearn.metrics import accuracy_score as accuracy_sklearn
from sklearn.metrics import precision_score as precision_sklearn
from sklearn.metrics import recall_score as recall_sklearn
from sklearn.metrics import precision_recall_fscore_support as prf_sklearn
from sklearn.exceptions import UndefinedMetricWarning

import warnings

from seqeval.metrics import precision_score as precision_seqeval
from seqeval.metrics import recall_score as recall_seqeval
from seqeval.metrics import f1_score as f1_seqeval
from seqeval.scheme import IOB2, BILOU

from nerblackbox.modules.ner_training.annotation_tags.tags import Tags


class NerMetrics:
    """
    On the token  level, the tags are evaluated in the given annotation scheme (e.g. plain, BIO)
    On the entity level, the tags are evaluated in the BIO scheme (after converting if needed)
    """

    def __init__(
        self,
        true_flat,
        pred_flat,
        level,
        scheme,
        classes=None,
        class_index=None,
        verbose=False,
    ):
        """
        :param true_flat:   [np array] of shape [batch_size * seq_length]
        :param pred_flat:   [np array] of shape [batch_size * seq_length]
        :param level:       [str] 'token' or 'entity'
        :param scheme:      [str] e.g. 'plain', 'bio'
        :param classes:     [optional, list] of [str] labels to take into account for metrics -> if level = 'token'
        :param class_index: [optional, int]            index to take into account for metrics -> if level = 'entity'
        :param verbose:     [optional, bool] if True, show verbose output
        """
        self.true_flat = true_flat  # token -> plain. entity -> plain, bio, bilou
        self.pred_flat = pred_flat  # token -> plain. entity -> plain, bio, bilou
        self.scheme = scheme  # token -> plain. entity -> plain, bio, bilou
        self.classes = classes
        self.class_index = class_index
        self.level = level
        self.verbose = verbose

        if self.scheme == "bilou":
            self.scheme_entity = "bilou"
            self.scheme_entity_seqeval = BILOU
        else:  # plain, bio
            self.scheme_entity = "bio"
            self.scheme_entity_seqeval = IOB2

        self.results = Results()
        self.failure_value = -1

        assert self.level in [
            "token",
            "entity",
        ], f"ERROR! level = {self.level} unknown."
        if self.level == "entity":
            self.true_flat_bio: List[str] = Tags(self.true_flat,).convert_scheme(
                source_scheme=self.scheme, target_scheme=self.scheme_entity
            )  # entity -> bio, bilou

            self.pred_flat_bio: List[str] = Tags(self.pred_flat).convert_scheme(
                source_scheme=self.scheme, target_scheme=self.scheme_entity
            )  # entity -> bio, bilou

            # ASR
            self.pred_flat_bio_corrected: List[str]
            self.pred_flat_bio_corrected, self.results.asr_abidance = Tags(
                self.pred_flat_bio
            ).restore_annotation_scheme_consistency(
                scheme=self.scheme_entity
            )  # entity -> bio, bilou

    def results_as_dict(self):
        return asdict(self.results)

    def compute(self, _metrics):
        """
        computes selected metrics
        ----------------------------------------------------------
        :param _metrics: [list] of [str], e.g. ['acc, 'precision']
        :return: -
        """
        warnings.filterwarnings("error")

        if "acc" in _metrics:
            self.accuracy()

        if "precision" in _metrics or "recall" in _metrics or "f1" in _metrics:
            self._compute_well_defined_classes()
            if "precision" in _metrics or "f1" in _metrics:
                self.precision()
            if "recall" in _metrics or "f1" in _metrics:
                self.recall()
            if "f1" in _metrics:
                self.f1_score()

        if (
            "asr_abidance" in _metrics
            or "asr_precision" in _metrics
            or "asr_recall" in _metrics
            or "asr_f1" in _metrics
        ):
            self.compute_asr_results()

        warnings.resetwarnings()

    def accuracy(self):
        """
        computes accuracy of predictions (_np_logits) w.r.t. ground truth (_np_label_ids)
        ---------------------------------------------------------------------------------
        :return: acc [np float]
        """
        self.results.acc = accuracy_sklearn(
            self.true_flat, self.pred_flat, normalize=True
        )

    def precision(self):
        """
        computes precision (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)

        Returns:
            precision_micro [np array] for all examples
            precision_macro [np array] for each class, then averaged
        """
        if self.level == "token":
            self.results.precision_micro = self._token_evaluation(
                evaluation_function=precision_sklearn, average="micro"
            )
            self.results.precision_macro = self._token_evaluation(
                evaluation_function=precision_sklearn, average="macro"
            )

        elif self.level == "entity":
            self.results.precision_micro = self._entity_evaluation_micro(
                evaluation_function=precision_seqeval
            )
            self.results.precision_macro = self._entity_evaluation_macro(
                evaluation_function=precision_seqeval,
            )

    def recall(self):
        """
        computes recall (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)

        Returns:
            recall_micro [np array] for all examples
            recall_macro [np array] for each class, then averaged
        """
        if self.level == "token":
            self.results.recall_micro = self._token_evaluation(
                evaluation_function=recall_sklearn, average="micro"
            )
            self.results.recall_macro = self._token_evaluation(
                evaluation_function=recall_sklearn, average="macro"
            )
        elif self.level == "entity":
            self.results.recall_micro = self._entity_evaluation_micro(
                evaluation_function=recall_seqeval
            )
            self.results.recall_macro = self._entity_evaluation_macro(
                evaluation_function=recall_seqeval
            )

    def f1_score(self):
        """
        computes f1 score (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)

        Returns:
            f1_score_micro [np array] for all examples
            f1_score_macro [np array] for each class, then averaged
        """
        if self.level == "token":
            self.results.f1_micro = self._token_evaluation(
                evaluation_function=prf_sklearn, average="micro"
            )
            self.results.f1_macro = self._token_evaluation(
                evaluation_function=prf_sklearn, average="macro"
            )
        elif self.level == "entity":
            self.results.f1_micro, self.results.f1_macro = self._entity_evaluation_f1(
                evaluation_function=f1_seqeval,
            )

    def compute_asr_results(self):
        """
        computes
        - self.results.asr_precision_micro
        - self.results.asr_recall_micro
        - self.results.asr_f1_micro
        """

        def _entity_evaluation_micro_asr(evaluation_function: Callable) -> float:
            """helper function"""
            try:
                metric = evaluation_function(
                    [self.true_flat_bio],
                    [self.pred_flat_bio_corrected],  # corrected !!!
                    average="micro",
                    mode="strict",
                    scheme=self.scheme_entity_seqeval,
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                metric = self.failure_value
            return metric

        self.results.asr_precision_micro = _entity_evaluation_micro_asr(
            evaluation_function=precision_seqeval
        )
        self.results.asr_recall_micro = _entity_evaluation_micro_asr(
            evaluation_function=recall_seqeval
        )
        self.results.asr_f1_micro = _entity_evaluation_micro_asr(
            evaluation_function=f1_seqeval
        )

    def _token_evaluation(self, evaluation_function: Callable, average: str) -> float:
        """
        compute precision/recall/f1 on token level

        Args:
            evaluation_function: precision_sklearn, recall_sklearn, prf_sklearn
            average: 'micro' or 'macro'

        Returns:
            metric: precision/recall on token level, 'micro' or 'macro' averaged
        """
        assert evaluation_function in [
            precision_sklearn,
            recall_sklearn,
            prf_sklearn,
        ], f"evaluation function = {evaluation_function} unknown / not allowed."
        assert average in ["micro", "macro"], f"average = {average} unknown."

        if self.classes is None or len(self.classes) > 1:  # "all" / "fil"
            if evaluation_function != prf_sklearn:
                metric = evaluation_function(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.classes,
                    average=average,
                    zero_division=0,
                )
            else:
                _, _, metric, _ = prf_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.classes,
                    average=average,
                    zero_division=0,
                )
        else:
            try:
                if evaluation_function != prf_sklearn:
                    metric = evaluation_function(
                        self.true_flat,
                        self.pred_flat,
                        labels=self.classes,
                        average=average,
                        zero_division="warn",
                    )
                else:
                    _, _, metric, _ = prf_sklearn(
                        self.true_flat,
                        self.pred_flat,
                        labels=self.classes,
                        average=average,
                        warn_for=("precision", "recall", "f-score"),
                        zero_division="warn",
                    )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                metric = self.failure_value

        return metric

    def _entity_evaluation_micro(self, evaluation_function: Callable) -> float:
        """
        compute precision/recall micro average on entity level

        Args:
            evaluation_function: precision_seqeval, recall_seqeval

        Returns:
            metric: precision/recall on entity level, 'macro' averaged
        """
        assert evaluation_function in [
            precision_seqeval,
            recall_seqeval,
        ], f"evaluation function = {evaluation_function} unknown / not allowed."

        if self.class_index is None:  # "fil"
            try:
                metric = evaluation_function(
                    [self.true_flat_bio],
                    [self.pred_flat_bio],
                    average="micro",
                    mode="strict",
                    scheme=self.scheme_entity_seqeval,
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                metric = self.failure_value
        else:  # "ind"
            try:
                metric = evaluation_function(
                    [self.true_flat_bio],
                    [self.pred_flat_bio],
                    mode="strict",
                    scheme=self.scheme_entity_seqeval,
                    average=None,
                    zero_division="warn",
                )[self.class_index]
            except UndefinedMetricWarning:
                try:
                    metric = evaluation_function(
                        [self.true_flat_bio],
                        [self.pred_flat_bio],
                        mode="strict",
                        scheme=self.scheme_entity_seqeval,
                        average=None,
                        zero_division=0,
                    )[self.class_index]
                except IndexError:
                    metric = self.failure_value

                if metric == 0:
                    metric = evaluation_function(
                        [self.true_flat_bio],
                        [self.pred_flat_bio],
                        mode="strict",
                        scheme=self.scheme_entity_seqeval,
                        average=None,
                        zero_division=1,
                    )[self.class_index]
                    if metric == 1:
                        metric = self.failure_value
            except IndexError:
                metric = self.failure_value

        return metric

    def _compute_well_defined_classes(self) -> None:
        """
        Created Attributes:
            results.classindices_macro: list of indices of well-defined classes in terms of precision, recall, f1
            results.numberofclasses_macro: number of well-defined classes in terms of precision, recall, f1
        """

        def _get_index_list(
            evaluation_function: Callable, true_array, pred_array, scheme_seqeval=None
        ):
            kwargs = (
                {"mode": "strict", "scheme": scheme_seqeval}
                if scheme_seqeval is not None
                else {}
            )
            try:
                metric_list = evaluation_function(
                    true_array,
                    pred_array,
                    average=None,
                    zero_division="warn",
                    **kwargs,
                )
                index_list = [i for i in range(len(metric_list))]
            except UndefinedMetricWarning:
                metric_list_all = evaluation_function(
                    true_array,
                    pred_array,
                    average=None,
                    zero_division=0,
                    **kwargs,
                )

                index_list = list()
                for index, metric_elem in enumerate(metric_list_all):
                    if metric_elem != 0:
                        index_list.append(index)
                    else:
                        metric_elem_alt = evaluation_function(
                            true_array,
                            pred_array,
                            average=None,
                            zero_division=1,
                            **kwargs,
                        )[index]
                        if metric_elem_alt != 1:
                            index_list.append(index)

            return index_list

        if self.level == "token":
            index_list_precision = _get_index_list(
                evaluation_function=precision_sklearn,
                true_array=self.true_flat,
                pred_array=self.pred_flat,
            )
            index_list_recall = _get_index_list(
                evaluation_function=recall_sklearn,
                true_array=self.true_flat,
                pred_array=self.pred_flat,
            )
        else:
            index_list_precision = _get_index_list(
                evaluation_function=precision_seqeval,
                true_array=[self.true_flat_bio],
                pred_array=[self.pred_flat_bio],
                scheme_seqeval=self.scheme_entity_seqeval,
            )
            index_list_recall = _get_index_list(
                evaluation_function=recall_seqeval,
                true_array=[self.true_flat_bio],
                pred_array=[self.pred_flat_bio],
                scheme_seqeval=self.scheme_entity_seqeval,
            )

        self.results.classindices_macro = tuple(
            [index for index in index_list_precision if index in index_list_recall]
        )
        if self.level == "token":
            self.results.numberofclasses_macro = (
                len(self.results.classindices_macro) - 1
            )  # disregard "O" label
        else:
            self.results.numberofclasses_macro = len(self.results.classindices_macro)

    def _entity_evaluation_macro(
        self,
        evaluation_function: Callable,
    ) -> float:
        """
        compute precision/recall macro average on entity level

        Args:
            evaluation_function: precision_seqeval, recall_seqeval

        Returns:
            metric: precision/recall on entity level, 'macro' averaged on well-defined classes
        """
        assert evaluation_function in [
            precision_seqeval,
            recall_seqeval,
        ], f"evaluation function = {evaluation_function} unknown / not allowed."

        metric = evaluation_function(
            [self.true_flat_bio],
            [self.pred_flat_bio],
            mode="strict",
            scheme=self.scheme_entity_seqeval,
            average="macro",
            zero_division=0,
        )
        return metric

    def _entity_evaluation_f1(
        self, evaluation_function: Callable
    ) -> Tuple[float, float]:
        """
        compute f1 micro or macro average on entity level

        Args:
            evaluation_function: f1_seqeval

        Returns:
            f1_micro: f1 on entity level, 'micro' averaged
            f1_macro: f1 on entity level, 'macro' averaged on well-defined classes
        """
        assert evaluation_function in [
            f1_seqeval
        ], f"evaluation function = {evaluation_function} unknown / not allowed."

        # ensure that precision and recall have been called:
        # self.precision()
        # self.recall()

        # f1_micro
        if (
            self.results.precision_micro == self.failure_value
            or self.results.recall_micro == self.failure_value
        ):
            f1_micro = self.failure_value
        else:
            if self.class_index is None:  # "fil"
                f1_micro = evaluation_function(
                    [self.true_flat_bio],
                    [self.pred_flat_bio],
                    average="micro",
                    mode="strict",
                    scheme=self.scheme_entity_seqeval,
                )
            else:  # "ind"
                f1_micro = evaluation_function(
                    [self.true_flat_bio],
                    [self.pred_flat_bio],
                    mode="strict",
                    scheme=self.scheme_entity_seqeval,
                    average=None,
                    zero_division="warn",
                )[self.class_index]

        # f1_macro
        if (
            self.results.precision_macro == self.failure_value
            or self.results.recall_macro == self.failure_value
        ):
            f1_macro = self.failure_value
        else:
            if self.class_index is None:  # "fil"
                metric_list = evaluation_function(
                    [self.true_flat_bio],
                    [self.pred_flat_bio],
                    mode="strict",
                    scheme=self.scheme_entity_seqeval,
                    average=None,
                )
                f1_macro = np.average(metric_list)
            else:  # "ind"
                f1_macro = self.failure_value

        return f1_micro, f1_macro


@dataclass
class Results:
    acc: float = -1
    precision_micro: float = -1
    precision_macro: float = -1
    recall_micro: float = -1
    recall_macro: float = -1
    f1_micro: float = -1
    f1_macro: float = -1
    classindices_macro: Tuple[float, ...] = ()
    numberofclasses_macro: float = -1
    asr_abidance: float = -1
    asr_precision_micro: float = -1
    asr_recall_micro: float = -1
    asr_f1_micro: float = -1
