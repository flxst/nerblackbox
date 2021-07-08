import pytest
from typing import List
from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics


class TestLoggedMetrics:

    logged_metrics = LoggedMetrics()

    @pytest.mark.parametrize(
        "required_tag_groups, "
        "required_phases, "
        "required_levels, "
        "required_averaging_groups, "
        "exclude, "
        "metrics",
        [
            (
                ["all"],
                ["train"],
                None,
                None,
                ["numberofclasses"],
                ["loss", "acc"],
            ),
            (
                ["all"],
                ["train"],
                None,
                None,
                ["numberofclasses", "acc"],
                ["loss"],
            ),
            (
                ["ind"],
                ["test"],
                None,
                None,
                ["numberofclasses"],
                ["precision", "recall", "f1"],
            ),
            (
                ["ind"],
                ["test"],
                None,
                ["micro"],
                ["numberofclasses"],
                ["precision", "recall", "f1"],
            ),
            (
                ["ind"],
                ["val"],
                None,
                ["micro"],
                ["numberofclasses"],
                [],
            ),
            (
                ["ind"],
                ["test"],
                None,
                ["macro"],
                ["numberofclasses"],
                [],
            ),
            (
                None,
                None,
                ["entity"],
                None,
                ["numberofclasses"],
                ["precision", "recall", "f1"],
            ),
            (
                None,
                None,
                None,
                None,
                ["numberofclasses"],
                ["loss", "acc", "precision", "recall", "f1"],
            ),
            (
                None,
                None,
                None,
                None,
                None,
                ["loss", "acc", "precision", "recall", "f1", "numberofclasses"],
            ),
            (
                None,
                None,
                ["entity"],
                ["macro"],
                None,
                ["precision", "recall", "f1", "numberofclasses"],
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
        assert set(test_metrics) == set(
            metrics
        ), f"test_metrics = {test_metrics} != {metrics}"

    @pytest.mark.parametrize(
        "metrics_flat_list",
        [
            (
                [
                    "entity_fil_numberofclasses_macro",
                    "entity_fil_precision_micro",
                    "entity_fil_precision_macro",
                    "entity_fil_recall_micro",
                    "entity_fil_recall_macro",
                    "entity_fil_f1_micro",
                    "entity_fil_f1_macro",
                    #
                    "entity_ind_precision_micro",
                    "entity_ind_recall_micro",
                    "entity_ind_f1_micro",
                    #
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
                    "token_fil_numberofclasses_macro",
                    "token_fil_precision_micro",
                    "token_fil_precision_macro",
                    "token_fil_recall_micro",
                    "token_fil_recall_macro",
                    "token_fil_f1_micro",
                    "token_fil_f1_macro",
                    #
                    "token_ind_precision_micro",
                    "token_ind_recall_micro",
                    "token_ind_f1_micro",
                    #
                    "token_O_precision_micro",
                    "token_O_recall_micro",
                    "token_O_f1_micro",
                ]
            )
        ],
    )
    def test_as_flat_list(self, metrics_flat_list: List[str]):
        test_metrics_flat_list = self.logged_metrics.as_flat_list()
        assert set(test_metrics_flat_list) == set(
            metrics_flat_list
        ), f"test_metrics_flat_list = {test_metrics_flat_list} != {metrics_flat_list}"
