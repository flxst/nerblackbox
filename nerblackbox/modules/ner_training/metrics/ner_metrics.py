from dataclasses import dataclass
from dataclasses import asdict
from typing import List, Optional, Tuple

from sklearn.metrics import accuracy_score as accuracy_sklearn
from sklearn.metrics import precision_score as precision_sklearn
from sklearn.metrics import recall_score as recall_sklearn
from sklearn.metrics import precision_recall_fscore_support as prf_sklearn
from sklearn.exceptions import UndefinedMetricWarning

import warnings

from seqeval.metrics import precision_score as precision_seqeval
from seqeval.metrics import recall_score as recall_seqeval
from seqeval.metrics import f1_score as f1_seqeval


class NerMetrics:
    def __init__(
        self,
        true_flat,
        pred_flat,
        tag_list=None,
        tag_index=None,
        level="token",
        plain_tags=False,
        verbose=False,
    ):
        """
        :param true_flat: [np array] of shape [batch_size * seq_length]
        :param pred_flat: [np array] of shape [batch_size * seq_length]
        :param tag_list:  [optional, list] of [str] labels to take into account for metrics -> if level = 'token'
        :param tag_index: [optional, int]            index to take into account for metrics -> if level = 'entity'
        :param level:     [optional, str] 'token' or 'entity'
        :param verbose:   [optional, bool] if True, show verbose output
        """
        self.true_flat = true_flat
        self.pred_flat = pred_flat
        self.tag_list = tag_list
        self.tag_index = tag_index
        self.level = level
        self.verbose = verbose

        self.results = Results()
        self.failure_value = -1

        assert self.level in ["token", "entity"], f"ERROR! level = {self.level} unknown."
        if self.level == "entity":
            self.true_flat_bio = convert_to_chunk(self.true_flat, to_bio=plain_tags)
            self.pred_flat_bio = convert_to_chunk(self.pred_flat, to_bio=plain_tags)

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
        if "precision" in _metrics:
            self.precision()
        if "recall" in _metrics:
            self.recall()
        if "f1" in _metrics:
            self.f1_score()

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
        -----------------------------------------------------------------------------------------------
        :return: precision_macro [np array] for each class, then averaged
                 precision_micro [np array] for all examples
        """
        if self.level == "token":
            self.results.precision_macro = self._token_evaluation(evaluation_function=precision_sklearn,
                                                                  average="macro")
            self.results.precision_micro = self._token_evaluation(evaluation_function=precision_sklearn,
                                                                  average="micro")

        elif self.level == "entity":
            self.results.precision_macro = self._entity_evaluation_macro(evaluation_function=precision_seqeval)
            self.results.precision_micro = self._entity_evaluation_micro(evaluation_function=precision_seqeval)

    def recall(self):
        """
        computes recall (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: recall_macro [np array] for each class, then averaged
                 recall_micro [np array] for all examples
        """
        if self.level == "token":
            self.results.recall_macro = self._token_evaluation(evaluation_function=recall_sklearn,
                                                               average="macro")
            self.results.recall_micro = self._token_evaluation(evaluation_function=recall_sklearn,
                                                               average="micro")
        elif self.level == "entity":
            self.results.recall_macro = self._entity_evaluation_macro(evaluation_function=recall_seqeval)
            self.results.recall_micro = self._entity_evaluation_micro(evaluation_function=recall_seqeval)

    def f1_score(self):
        """
        computes f1 score (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: f1_score_macro [np array] for each class, then averaged
                 f1_score_micro [np array] for all examples
        """
        if self.level == "token":
            self.results.f1_macro = self._token_evaluation(evaluation_function=prf_sklearn,
                                                           average="macro")
            self.results.f1_micro = self._token_evaluation(evaluation_function=prf_sklearn,
                                                           average="micro")
        elif self.level == "entity":
            self.results.f1_macro, self.results.f1_micro = self._entity_evaluation_f1(evaluation_function=f1_seqeval)

    def _token_evaluation(self, evaluation_function: callable, average: str) -> float:
        """
        compute precision/recall/f1 on token level

        Args:
            evaluation_function: precision_sklearn, recall_sklearn, prf_sklearn
            average: 'micro' or 'macro'

        Returns:
            metric: precision/recall on token level, 'micro' or 'macro' averaged
        """
        assert evaluation_function in [precision_sklearn, recall_sklearn, prf_sklearn], \
            f"evaluation function = {evaluation_function} unknown / not allowed."
        assert average in ["micro", "macro"], f"average = {average} unknown."

        try:
            if evaluation_function != prf_sklearn:
                metric = evaluation_function(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average=average,
                )
            else:
                _, _, metric, _ = prf_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average=average,
                    warn_for=("precision", "recall", "f-score"),
                )
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            metric = self.failure_value

        return metric

    def _entity_evaluation_macro(self, evaluation_function: callable) -> float:
        """
        compute precision/recall macro average on entity level

        Args:
            evaluation_function: precision_seqeval, recall_seqeval

        Returns:
            metric: precision/recall on entity level, 'macro' averaged
        """
        assert evaluation_function in [precision_seqeval, recall_seqeval], \
            f"evaluation function = {evaluation_function} unknown / not allowed."

        try:
            metric = evaluation_function(
                [self.true_flat_bio], [self.pred_flat_bio], average="macro"
            )
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            metric = self.failure_value

        return metric

    def _entity_evaluation_micro(self, evaluation_function: callable) -> float:
        """
        compute precision/recall micro average on entity level

        Args:
            evaluation_function: precision_seqeval, recall_seqeval

        Returns:
            metric: precision/recall on entity level, 'macro' averaged
        """
        assert evaluation_function in [precision_seqeval, recall_seqeval], \
            f"evaluation function = {evaluation_function} unknown / not allowed."

        if self.tag_index is None:  # "fil"
            try:
                metric = evaluation_function(
                    [self.true_flat_bio], [self.pred_flat_bio], average="micro"
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                metric = self.failure_value
        else:  # "ind"
            try:
                metric = evaluation_function(
                    [self.true_flat_bio], [self.pred_flat_bio], average=None, zero_division="warn",
                )[self.tag_index]
            except UndefinedMetricWarning:
                try:
                    metric = evaluation_function(
                        [self.true_flat_bio], [self.pred_flat_bio], average=None, zero_division=0
                    )[self.tag_index]
                except IndexError:
                    metric = self.failure_value

                if metric == 0:
                    metric = evaluation_function(
                        [self.true_flat_bio], [self.pred_flat_bio], average=None, zero_division=1
                    )[self.tag_index]
                    if metric == 1:
                        metric = self.failure_value
            except IndexError:
                metric = self.failure_value

        return metric

    def _entity_evaluation_f1(self, evaluation_function: callable) -> Tuple[float, float]:
        """
        compute f1 micro or macro average on entity level

        Args:
            evaluation_function: f1_seqeval

        Returns:
            f1_macro: f1 on entity level, 'macro' averaged
            f1_micro: f1 on entity level, 'macro' averaged
        """
        assert evaluation_function in [f1_seqeval], \
            f"evaluation function = {evaluation_function} unknown / not allowed."

        self.precision()
        self.recall()

        # f1_macro
        if self.results.precision_macro == self.failure_value or \
                self.results.recall_macro == self.failure_value:
            f1_macro = self.failure_value
        else:
            if self.tag_index is None:  # "fil"
                f1_macro = evaluation_function(
                    [self.true_flat_bio], [self.pred_flat_bio], average="macro"
                )
            else:  # "ind"
                f1_macro = self.failure_value

        # f1_micro
        if self.results.precision_micro == self.failure_value or \
                self.results.recall_micro == self.failure_value:
            f1_micro = self.failure_value
        else:
            if self.tag_index is None:  # "fil"
                f1_micro = evaluation_function(
                    [self.true_flat_bio], [self.pred_flat_bio], average="micro"
                )
            else:  # "ind"
                f1_micro = evaluation_function(
                    [self.true_flat_bio], [self.pred_flat_bio], average=None, zero_division="warn",
                )[self.tag_index]

        return f1_macro, f1_micro


@dataclass
class Results:
    acc: float = -1
    precision_micro: float = -1
    precision_macro: float = -1
    recall_micro: float = -1
    recall_macro: float = -1
    f1_micro: float = -1
    f1_macro: float = -1


def convert_to_chunk(tag_list: List[str], to_bio=True) -> List[str]:
    """
    - get rid of special tokens
    - add bio prefixed to tags

    Args:
        tag_list:      e.g. ['O',   'ORG',   'ORG']
        to_bio:        whether to cast to bio labels

    Returns:
        bio_tag_list:  e.g. ['O', 'B-ORG', 'I-ORG']
    """
    clean_tag_list = get_rid_of_special_tokens(tag_list)
    if to_bio:
        assert_plain_tags(clean_tag_list)
        return add_bio_to_tag_list(clean_tag_list)
    else:
        assert_bio_tags(clean_tag_list)
        return clean_tag_list


def assert_plain_tags(tag_list: List[str]) -> None:
    for tag in tag_list:
        if tag != "O" and (len(tag) > 2 and tag[1] == "-"):
            raise Exception(
                "ERROR! attempt to convert tags to bio format that already seem to have bio format."
            )


def assert_bio_tags(tag_list: List[str]) -> None:
    for tag in tag_list:
        if tag != "O" and (len(tag) <= 2 or tag[1] != "-"):
            raise Exception(
                "ERROR! assuming tags to have bio format that seem to have plain format instead."
            )


def add_bio_to_tag_list(tag_list: List[str]) -> List[str]:
    """
    adds bio prefixes to tags

    Args:
        tag_list:     e.g. ['O',   'ORG',   'ORG']

    Returns:
        bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']
    """
    return [
        _add_bio_to_tag(tag_list[i], previous=tag_list[i - 1] if i > 0 else None)
        for i in range(len(tag_list))
    ]


def _add_bio_to_tag(tag: str, previous: Optional[str] = None) -> str:
    """
    add bio prefix to tag, depending on previous tag

    Args:
        tag:      e.g. 'ORG'
        previous: e.g. 'ORG'

    Returns:
        bio_tag:  e.g. 'I-ORG'
    """
    if tag == "O" or tag.startswith("["):
        return tag
    elif previous is None:
        return f"B-{tag}"
    elif tag != previous:
        return f"B-{tag}"
    else:
        return f"I-{tag}"


def get_rid_of_special_tokens(tag_list: List[str]) -> List[str]:
    """
    replace special tokens ('[CLS]', '[SEP]', '[PAD]') by 'O'

    Args:
        tag_list:         e.g. ['[CLS]', 'O', 'ORG', 'ORG', '[SEP]']

    Returns:
        cleaned_tag_list: e.g. [    'O', 'O', 'ORG', 'ORG',     'O']
    """

    return [tag if not tag.startswith("[") else "O" for tag in tag_list]
