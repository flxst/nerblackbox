import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from nerblackbox.modules.ner_training.metrics.ner_metrics import (
    NerMetrics,
)
from nerblackbox.tests.test_utils import pytest_approx
from typing import List


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestNerMetricsTable:

    metrics_simple = ["acc"]
    metrics_micro_macro = ["precision", "recall", "f1"]
    metrics_special = ["numberofclasses_macro"]
    metrics = metrics_simple + metrics_micro_macro

    df = {
        level: pd.read_csv(
            resource_filename(
                "nerblackbox", f"tests/test_data/test_ner_metrics_{level}.csv"
            ),
            sep=";",
        )
        for level in ["token", "entity"]
    }
    level: str
    labels: List[str]
    micro: bool
    macro: bool

    def test_predictions_from_csv_token(self):
        self.level = "token"
        self.labels = ["all", "fil", "A", "B", "C", "O"]
        self.micro = True
        self.macro = True
        self._test_predictions_from_csv()

    def test_predictions_from_csv_entity(self):
        self.level = "entity"
        self.labels = ["fil", "A", "B", "C"]
        self.micro = True
        self.macro = True
        self._test_predictions_from_csv()

    ####################################################################################################################
    # TEST #############################################################################################################
    ####################################################################################################################
    def _test_predictions_from_csv(self) -> None:
        """
        test true against pred values for all rows and labels
        """
        nr_rows = len(self.df[self.level])
        true = self._seq2array(self.df[self.level]["sequence"].iloc[0])

        for row in range(nr_rows):
            pred = self._seq2array(self.df[self.level]["sequence"].iloc[row])
            tested_columns = list()
            for labels in self.labels:
                _tested_columns = self._single_row_and_label_category_test(
                    true, pred, row=row, labels=labels
                )
                tested_columns.extend(_tested_columns)

            assert set(tested_columns) == set(self.df[self.level].columns[2:])

    def _single_row_and_label_category_test(
        self, true: np.array, pred: np.array, row: int, labels: str
    ) -> List[str]:
        """
        test true against pred values for single row in csv and specific labels

        Args:
            true:   true values, e.g. [A, A, O, O, B]
            pred:   pred values, e.g. [A, A, A, O, B]
            row:    e.g. 2
            labels: e.g. 'all', 'A', 'B'

        Returns:
            tested_metrics: labels_metrics, e.g. ['all-precision', 'B-recall', ..]
        """

        def get_classes(_labels):
            if _labels == "all":
                return None
            elif _labels == "fil":
                return ["A", "B", "C"]
            else:
                return _labels

        def get_class_index(_labels):
            if _labels == "fil":
                return None
            else:
                return ["A", "B", "C"].index(_labels)

        ner_metrics = NerMetrics(
            true,
            pred,
            level=self.level,
            scheme="plain",
            classes=get_classes(labels) if self.level == "token" else None,
            class_index=get_class_index(labels) if self.level == "entity" else None,
            verbose=True,
        )
        ner_metrics.compute(self.metrics)
        ner_metrics_results = ner_metrics.results_as_dict()

        metrics_extended = self._extend_all_metrics(labels)
        _tested_metrics = list()
        for metric in metrics_extended:
            labels_metric = f"{labels}-{metric}"

            ner_metrics_results_metric = ner_metrics_results[
                self._extend_metric(labels, metric)
            ]
            assert ner_metrics_results_metric == pytest_approx(
                self.df[self.level][labels_metric][row]
            ), f"{self.level}: pred_{row}, {labels_metric}"

            _tested_metrics.append(labels_metric)

        return _tested_metrics

    ####################################################################################################################
    # HELPERS ##########################################################################################################
    ####################################################################################################################
    def _extend_all_metrics(self, _labels):
        """
        derive _metrics_extended from self.metrics in case of labels == 'fil', 'all'
        ----------------------------------------------------------------------------
        :param _labels: [str], e.g. 'all', 'A', 'B'
        :return: _metrics_extended [list] of [str], e.g. ['acc', 'precision_micro', 'precision_macro', ..]
        """
        _metrics_extended = list()
        if _labels == "all":
            for field in self.metrics_simple:
                if field in self.metrics:
                    _metrics_extended.append(field)
            for field in self.metrics_micro_macro:
                if field in self.metrics:
                    if self.macro:
                        _metrics_extended.append(f"{field}_macro")
        elif _labels == "fil":
            for field in self.metrics_micro_macro:
                if field in self.metrics:
                    if self.macro:
                        _metrics_extended.append(f"{field}_macro")
                    if self.micro:
                        _metrics_extended.append(f"{field}_micro")
            for field in self.metrics_special:
                _metrics_extended.append(f"{field}")
        else:
            _metrics_extended = self.metrics_micro_macro
        return _metrics_extended

    def _extend_metric(self, _labels, _metric):
        """
        extend _metric with '_micro' suffix
        -----------------------------------
        :param _labels: [str], e.g. 'all', 'A', 'B'
        :param _metric: [str], e.g. 'acc', 'precision', 'recall', 'f1', ..
        :return: _extended_metric: [str], e.g. 'precision_micro'
        """
        return (
            _metric
            if (_labels in ["fil", "all"] or _metric in self.metrics_simple)
            else f"{_metric}_micro"
        )

    @staticmethod
    def _seq2array(_str):
        """
        convert sequence str to numpy array
        -----------------------------------
        :param _str: sequence str [str],      e.g. '[A, A, O, O, B]'
        :return:                  [np array], e.g. np.array(['A', 'A', 'O', 'O', 'B'])
        """
        return np.array(_str.strip("[").strip("]").replace(" ", "").split(","))
