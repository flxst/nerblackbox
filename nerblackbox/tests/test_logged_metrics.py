import pytest
from typing import List
from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics


class TestLoggedMetrics:

    logged_metrics = LoggedMetrics()

    @pytest.mark.parametrize(
        "required_tag_groups, " "required_phases, " "required_levels, " "required_averaging_groups, " "exclude, " "metrics",
        [
            (
                ["all"],
                ["train"],
                None,
                None,
                None,
                ["loss", "acc"],
            ),
            (
                ["all"],
                ["train"],
                None,
                None,
                ["acc"],
                ["loss"],
            ),
            (
                ["ind"],
                ["val"],
                None,
                None,
                None,
                ["f1", "precision", "recall"],
            ),
            (
                ["ind"],
                ["val"],
                None,
                ["micro"],
                None,
                ["f1", "precision", "recall"],
            ),
            (
                ["ind"],
                ["val"],
                None,
                ["macro"],
                None,
                [],
            ),
            (
                None,
                None,
                ["entity"],
                None,
                None,
                ["precision", "recall", "f1"],
            ),
            (
                None,
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
        required_levels: List[str],
        required_averaging_groups: List[str],
        exclude: List[str],
        metrics: List[str],
    ):

        test_metrics = self.logged_metrics.get_metrics(
            required_tag_groups=required_tag_groups,
            required_phases=required_phases,
            required_levels=required_levels,
            required_averaging_groups=required_averaging_groups,
            exclude=exclude,
        )
        assert set(test_metrics) == set(metrics), f"ERROR!"

    @pytest.mark.parametrize(
        "metrics_flat_list",
        [
            (
                [
                    "token_all_loss",
                    "token_all_acc",
                    #
                    "token_all_precision_micro",
                    "token_all_precision_macro",
                    "token_all_recall_micro",
                    "token_all_recall_macro",
                    "token_all_f1_micro",
                    "token_all_f1_macro",
                    #
                    "token_fil_precision_micro",
                    "token_fil_precision_macro",
                    "token_fil_recall_micro",
                    "token_fil_recall_macro",
                    "token_fil_f1_micro",
                    "token_fil_f1_macro",
                    #
                    "entity_chk_precision_micro",
                    "entity_chk_recall_micro",
                    "entity_chk_f1_micro",
                    #
                    "token_ind_precision_micro",
                    "token_ind_recall_micro",
                    "token_ind_f1_micro",
                ]
            )
        ],
    )
    def test_as_flat_list(self, metrics_flat_list: List[str]):
        test_metrics_flat_list = self.logged_metrics.as_flat_list()
        assert set(test_metrics_flat_list) == set(metrics_flat_list), f"ERROR! 2"
