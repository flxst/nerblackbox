import numpy as np
import pandas as pd
import pytest
from pkg_resources import resource_filename
from nerblackbox.modules.ner_training.metrics.ner_metrics import NerMetrics


class TestNerMetrics:

    labels = ["all", "fil", "A", "B", "O"]

    metrics_simple = ["acc"]
    metrics_micro_macro = ["precision", "recall", "f1"]
    metrics = metrics_simple + metrics_micro_macro

    path_csv = resource_filename("nerblackbox", "tests/test_ner_metrics.csv")
    df = pd.read_csv(path_csv, sep=";")

    ####################################################################################################################
    # TEST #############################################################################################################
    ####################################################################################################################
    def test_predictions_from_csv(self):
        """
        test true against pred values for all rows and labels
        -----------------------------------------------------
        :return: -
        """
        nr_rows = len(self.df)
        true = self._seq2array(self.df["sequence"].iloc[0])

        for row in range(nr_rows):
            pred = self._seq2array(self.df["sequence"].iloc[row])
            tested_columns = list()
            for labels in self.labels:
                _tested_columns = self._single_row_and_label_category_test(
                    true, pred, row=row, labels=labels
                )
                tested_columns.extend(_tested_columns)

            assert set(tested_columns) == set(self.df.columns[2:])

    def _single_row_and_label_category_test(self, true, pred, row, labels):
        """
        test true against pred values for single row in csv and specific labels
        -----------------------------------------------------------------------
        :param true:   [np array] with true values, e.g. [A, A, O, O, B]
        :param pred:   [np array] with pred values, e.g. [A, A, A, O, B]
        :param row:    [int], e.g. 2
        :param labels: [str], e.g. 'all', 'A', 'B'
        :return: tested_metrics [list] of [str] w/ labels_metrics, e.g. ['all-precision', 'B-recall', ..]
        """

        def get_tag_list(_labels):
            if labels == "all":
                return None
            elif labels == "fil":
                return ["A", "B"]
            else:
                return _labels

        ner_metrics = NerMetrics(
            true, pred, tag_list=get_tag_list(labels), level="token"
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
            assert ner_metrics_results_metric == self._pytest_approx(
                self.df[labels_metric][row]
            ), f"pred_{row}, {labels_metric}"

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
                    _metrics_extended.append(f"{field}_macro")
        elif _labels == "fil":
            for field in self.metrics_micro_macro:
                if field in self.metrics:
                    _metrics_extended.append(f"{field}_micro")
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

    @staticmethod
    def _pytest_approx(number):
        """
        get acceptable pytest range for number
        --------------------------------------
        :param number: [float], e.g. 0.82
        :return: pytest range, e.g. 0.82 +- 0.01
        """
        test_precision = 0.01
        return pytest.approx(number, abs=test_precision)


if __name__ == "__main__":
    test_ner_metrics = TestNerMetrics()
    test_ner_metrics.test_predictions_from_csv()
