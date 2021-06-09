from dataclasses import dataclass
from dataclasses import asdict
from typing import List

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
        level="token",
        plain_tags=False,
        verbose=False,
    ):
        """
        :param true_flat: [np array] of shape [batch_size * seq_length]
        :param pred_flat: [np array] of shape [batch_size * seq_length]
        :param tag_list:  [optional, list] of [str] labels to take into account for metrics
        :param level:     [optional, str] 'token' or 'chunk'
        :param verbose:   [optional, bool] if True, show verbose output
        """
        self.true_flat = true_flat
        self.pred_flat = pred_flat
        self.tag_list = tag_list
        self.level = level
        self.verbose = verbose

        self.results = Results()
        self.failure_value = -1

        if self.level == "chunk":
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
            # precision_macro
            try:
                self.results.precision_macro = precision_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average="macro",
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                self.results.precision_macro = self.failure_value

            # precision_micro
            try:
                self.results.precision_micro = precision_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average="micro",
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                self.results.precision_micro = self.failure_value
        else:
            self.results.precision_micro = precision_seqeval(
                self.true_flat_bio, self.pred_flat_bio, average="micro"
            )

    def recall(self):
        """
        computes recall (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: recall_macro [np array] for each class, then averaged
                 recall_micro [np array] for all examples
        """
        if self.level == "token":
            try:
                # recall_macro
                self.results.recall_macro = recall_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average="macro",
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                self.results.recall_macro = self.failure_value

            # recall_micro
            try:
                self.results.recall_micro = recall_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average="micro",
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                self.results.recall_micro = self.failure_value
        else:
            self.results.recall_micro = recall_seqeval(
                self.true_flat_bio, self.pred_flat_bio, average="micro"
            )

    def f1_score(self):
        """
        computes f1 score (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: f1_score_macro [np array] for each class, then averaged
                 f1_score_micro [np array] for all examples
        """
        if self.level == "token":
            try:
                # f1_macro
                _, _, self.results.f1_macro, _ = prf_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average="macro",
                    warn_for=("precision", "recall", "f-score"),
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                self.results.f1_macro = self.failure_value

            # f1_micro
            try:
                _, _, self.results.f1_micro, _ = prf_sklearn(
                    self.true_flat,
                    self.pred_flat,
                    labels=self.tag_list,
                    average="micro",
                    warn_for=("precision", "recall", "f-score"),
                )
            except UndefinedMetricWarning as e:
                if self.verbose:
                    print(e)
                self.results.f1_micro = self.failure_value
        else:
            self.results.f1_micro = f1_seqeval(
                self.true_flat_bio, self.pred_flat_bio, average="micro"
            )


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
            raise Exception("ERROR! attempt to convert tags to bio format that already seem to have bio format.")


def assert_bio_tags(tag_list: List[str]) -> None:
    for tag in tag_list:
        if tag != "O" and (len(tag) <= 2 or tag[1] != "-"):
            raise Exception("ERROR! assuming tags to have bio format that seem to have plain format instead.")


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


def _add_bio_to_tag(tag: str, previous: str) -> str:
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
