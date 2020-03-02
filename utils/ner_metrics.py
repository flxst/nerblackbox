
from dataclasses import dataclass
from dataclasses import asdict

from sklearn.metrics import accuracy_score as accuracy_sklearn
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import precision_score as precision_sklearn
from sklearn.metrics import recall_score as recall_sklearn


@dataclass
class Results:
    acc: float = None
    precision_micro: float = None
    precision_macro: float = None
    recall_micro: float = None
    recall_macro: float = None
    f1_micro: float = None
    f1_macro: float = None


class NerMetrics:

    def __init__(self,
                 true_flat,
                 pred_flat,
                 labels=None):
        """
        :param true_flat: [np array] of shape [batch_size * seq_length]
        :param pred_flat: [np array] of shape [batch_size * seq_length]
        :param labels: [optional, list] of [str] labels to take into account for metrics
        """
        self.true_flat = true_flat
        self.pred_flat = pred_flat
        self.labels = labels

        self.results = Results()

    def results_as_dict(self):
        return asdict(self.results)

    def compute(self, _metrics):
        if 'acc' in _metrics:
            self.accuracy()
        if 'precision' in _metrics:
            self.precision()
        if 'recall' in _metrics:
            self.recall()
        if 'f1' in _metrics:
            self.f1_score()

    def accuracy(self):
        """
        computes accuracy of predictions (_np_logits) w.r.t. ground truth (_np_label_ids)
        ---------------------------------------------------------------------------------
        :return: acc [np float]
        """
        self.results.acc = accuracy_sklearn(self.pred_flat, self.true_flat, normalize=True)

    def precision(self):
        """
        computes precision (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: precision_macro [np array] for each class, then averaged
                 precision_micro [np array] for all examples
        """
        self.results.precision_macro = precision_sklearn(self.true_flat, self.pred_flat, labels=self.labels, average='macro')
        self.results.precision_micro = precision_sklearn(self.true_flat, self.pred_flat, labels=self.labels, average='micro')

    def recall(self):
        """
        computes recall (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: recall_macro [np array] for each class, then averaged
                 recall_micro [np array] for all examples
        """
        self.results.recall_macro = recall_sklearn(self.true_flat, self.pred_flat, labels=self.labels, average='macro')
        self.results.recall_micro = recall_sklearn(self.true_flat, self.pred_flat, labels=self.labels, average='micro')

    def f1_score(self):
        """
        computes f1 score (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :return: f1_score_macro [np array] for each class, then averaged
                 f1_score_micro [np array] for all examples
        """
        self.results.f1_macro = f1_score_sklearn(self.true_flat, self.pred_flat, labels=self.labels, average='macro')
        self.results.f1_micro = f1_score_sklearn(self.true_flat, self.pred_flat, labels=self.labels, average='micro')
