import pytest
from typing import List
from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics


class TestLoggedMetrics:

    logged_metrics = LoggedMetrics()

    @pytest.mark.parametrize(
        "required_tag_groups, " "required_phases, " "required_averaging_groups, " "exclude, " "metrics",
        [
            (
                ["all"],
                ["train"],
                None,
                None,
                ["loss", "acc"],
            ),
            (
                ["all"],
                ["train"],
                None,
                ["acc"],
                ["loss"],
            ),
            (
                ["ind"],
                ["val"],
                None,
                None,
                ["f1", "precision", "recall"],
            ),
            (
                ["ind"],
                ["val"],
                ["micro"],
                None,
                ["f1", "precision", "recall"],
            ),
            (
                ["ind"],
                ["val"],
                ["macro"],
                None,
                [],
            ),
            (
                None,
                None,
                None,
                None,
                ["loss", "precision", "acc", "recall", "f1"],
            ),
        ],
    )
    def test_get_metrics(
        self,
        required_tag_groups: List[str],
        required_phases: List[str],
        required_averaging_groups: List[str],
        exclude: List[str],
        metrics: List[str],
    ):

        test_metrics = self.logged_metrics.get_metrics(
            required_tag_groups=required_tag_groups,
            required_phases=required_phases,
            required_averaging_groups=required_averaging_groups,
            exclude=exclude,
        )
        assert set(test_metrics) == set(metrics), f"ERROR!"

    @pytest.mark.parametrize(
        "metrics_flat_list",
        [
            (
                [
                    "all_loss",
                    "all_acc",
                    #
                    "all_precision_micro",
                    "all_precision_macro",
                    "all_recall_micro",
                    "all_recall_macro",
                    "all_f1_micro",
                    "all_f1_macro",
                    #
                    "fil_precision_micro",
                    "fil_precision_macro",
                    "fil_recall_micro",
                    "fil_recall_macro",
                    "fil_f1_micro",
                    "fil_f1_macro",
                    #
                    "chk_precision_micro",
                    "chk_recall_micro",
                    "chk_f1_micro",
                    #
                    "ind_precision_micro",
                    "ind_recall_micro",
                    "ind_f1_micro",
                ]
            )
        ],
    )
    def test_as_flat_list(self, metrics_flat_list: List[str]):
        test_metrics_flat_list = self.logged_metrics.as_flat_list()
        assert set(test_metrics_flat_list) == set(metrics_flat_list), f"ERROR! 2"
