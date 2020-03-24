
from dataclasses import dataclass
from dataclasses import asdict

from sklearn.metrics import accuracy_score as accuracy_sklearn
from sklearn.metrics import precision_score as precision_sklearn
from sklearn.metrics import recall_score as recall_sklearn
from sklearn.metrics import precision_recall_fscore_support as prf_sklearn
from sklearn.exceptions import UndefinedMetricWarning

import warnings


@dataclass
class Results:
    acc: float = -1
    precision_micro: float = -1
    precision_macro: float = -1
    recall_micro: float = -1
    recall_macro: float = -1
    f1_micro: float = -1
    f1_macro: float = -1


@dataclass
class Failures:
    acc: int = 0
    precision_micro: int = 0
    precision_macro: int = 0
    recall_micro: int = 0
    recall_macro: int = 0
    f1_micro: int = 0
    f1_macro: int = 0


class NerMetrics:

    def __init__(self,
                 true_flat,
                 pred_flat,
                 tag_list=None,
                 verbose=False):
        """
        :param true_flat: [np array] of shape [batch_size * seq_length]
        :param pred_flat: [np array] of shape [batch_size * seq_length]
        :param tag_list:  [optional, list] of [str] labels to take into account for metrics
        :param verbose:   [optional, bool] if True, show verbose output
        """
        self.true_flat = true_flat
        self.pred_flat = pred_flat
        self.tag_list = tag_list
        self.verbose = verbose

        self.results = Results()
        self.failures = Failures()
        self.failure_value = -1

    def results_as_dict(self):
        return asdict(self.results)

    def failures_as_dict(self):
        return asdict(self.failures)

    def compute(self, _metrics):
        """
        computes selected metrics
        ----------------------------------------------------------
        :param _metrics: [list] of [str], e.g. ['acc, 'precision']
        :return: -
        """
        warnings.filterwarnings("error")

        if 'acc' in _metrics:
            self.accuracy()
        if 'precision' in _metrics:
            self.precision()
        if 'recall' in _metrics:
            self.recall()
        if 'f1' in _metrics:
            self.f1_score()

        warnings.resetwarnings()

    def accuracy(self):
        """
        computes accuracy of predictions (_np_logits) w.r.t. ground truth (_np_label_ids)
        ---------------------------------------------------------------------------------
        :return: acc [np float]
        """
        self.results.acc = accuracy_sklearn(self.true_flat, self.pred_flat, normalize=True)

    def precision(self):
        """
        computes precision (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: precision_macro [np array] for each class, then averaged
                 precision_micro [np array] for all examples
        """
        try:
            self.results.precision_macro = precision_sklearn(self.true_flat, self.pred_flat, labels=self.tag_list, average='macro')
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            self.results.precision_macro = self.failure_value
            self.failures.precision_macro += 1

        try:
            self.results.precision_micro = precision_sklearn(self.true_flat, self.pred_flat, labels=self.tag_list, average='micro')
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            self.results.precision_macro = self.failure_value
            self.failures.precision_micro += 1

    def recall(self):
        """
        computes recall (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: recall_macro [np array] for each class, then averaged
                 recall_micro [np array] for all examples
        """
        try:
            self.results.recall_macro = recall_sklearn(self.true_flat, self.pred_flat, labels=self.tag_list, average='macro')
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            self.results.precision_macro = self.failure_value
            self.failures.recall_macro += 1

        try:
            self.results.recall_micro = recall_sklearn(self.true_flat, self.pred_flat, labels=self.tag_list, average='micro')
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            self.results.precision_macro = self.failure_value
            self.failures.recall_micro += 1

    def f1_score(self):
        """
        computes f1 score (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: f1_score_macro [np array] for each class, then averaged
                 f1_score_micro [np array] for all examples
        """
        try:
            _, _, self.results.f1_macro, _ = prf_sklearn(self.true_flat, self.pred_flat, labels=self.tag_list, average='macro', warn_for=('precision', 'recall', 'f-score'))
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            self.results.precision_macro = self.failure_value
            self.failures.f1_macro += 1

        try:
            _, _, self.results.f1_micro, _ = prf_sklearn(self.true_flat, self.pred_flat, labels=self.tag_list, average='micro', warn_for=('precision', 'recall', 'f-score'))
        except UndefinedMetricWarning as e:
            if self.verbose:
                print(e)
            self.results.precision_macro = self.failure_value
            self.failures.f1_micro += 1
