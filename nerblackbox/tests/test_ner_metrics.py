import numpy as np
import pandas as pd
import pytest
from pkg_resources import resource_filename
from nerblackbox.modules.ner_training.metrics.ner_metrics import (
    NerMetrics,
    get_rid_of_special_tokens,
    convert_to_chunk,
)
from typing import List


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestNerMetricsTable:

    metrics_simple = ["acc"]
    metrics_micro_macro = ["precision", "recall", "f1"]
    metrics = metrics_simple + metrics_micro_macro

    df = {
        level: pd.read_csv(
            resource_filename(
                "nerblackbox", f"tests/test_data/test_ner_metrics_{level}.csv"
            ),
            sep=";",
        )
        for level in ["token", "chunk"]
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

    def test_predictions_from_csv_chunk(self):
        self.level = "chunk"
        self.labels = ["fil"]
        self.micro = True
        self.macro = False
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

        def get_tag_list(_labels):
            if labels == "all":
                return None
            elif labels == "fil":
                return ["A", "B"]
            else:
                return _labels

        ner_metrics = NerMetrics(
            true,
            pred,
            tag_list=get_tag_list(labels),
            level=self.level,
            plain_tags=True,
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
            assert ner_metrics_results_metric == self._pytest_approx(
                self.df[self.level][labels_metric][row]
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
                    if self.macro:
                        _metrics_extended.append(f"{field}_macro")
        elif _labels == "fil":
            for field in self.metrics_micro_macro:
                if field in self.metrics:
                    if self.macro:
                        _metrics_extended.append(f"{field}_macro")
                    if self.micro:
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


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestNerMetrics:
    @pytest.mark.parametrize(
        "input_sequence, " "output_sequence",
        [
            (
                ["A", "A", "O", "O", "B"],
                ["A", "A", "O", "O", "B"],
            ),
            (
                ["[CLS]", "A", "A", "O", "O", "B", "[SEP]"],
                ["O", "A", "A", "O", "O", "B", "O"],
            ),
            (
                ["[CLS]", "A", "A", "[UNK]", "O", "O", "B", "[SEP]"],
                ["O", "A", "A", "O", "O", "O", "B", "O"],
            ),
        ],
    )
    def test_get_rid_of_special_tokens(
        self, input_sequence: List[str], output_sequence: List[str]
    ):
        test_output_sequence = get_rid_of_special_tokens(input_sequence)
        assert (
            test_output_sequence == output_sequence
        ), f"{test_output_sequence} != {output_sequence}"

    @pytest.mark.parametrize(
        "input_sequence, " "to_bio, " "output_sequence",
        [
            (
                ["[CLS]", "A", "A", "[UNK]", "O", "O", "B", "[SEP]"],
                True,
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                ["[CLS]", "B-A", "I-A", "[UNK]", "O", "O", "B-B", "[SEP]"],
                False,
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                ["[CLS]", "A", "A", "[UNK]", "O", "O", "B", "[SEP]"],
                False,
                None,
            ),
            (
                ["[CLS]", "B-A", "I-A", "[UNK]", "O", "O", "B-B", "[SEP]"],
                True,
                None,
            ),
        ],
    )
    def test_convert_to_chunk(
        self, input_sequence: List[str], to_bio: bool, output_sequence: List[str]
    ):
        if output_sequence is not None:
            test_output_sequence = convert_to_chunk(input_sequence, to_bio)
            assert (
                test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                convert_to_chunk(input_sequence, to_bio)


if __name__ == "__main__":
    test_ner_metrics_table = TestNerMetricsTable()
    test_ner_metrics_table.test_predictions_from_csv_token()
    test_ner_metrics_table.test_predictions_from_csv_chunk()

    test_ner_metrics = TestNerMetrics()
    test_ner_metrics.test_get_rid_of_special_tokens()
    test_ner_metrics.test_convert_to_chunk()
