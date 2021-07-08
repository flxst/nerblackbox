from typing import List
from itertools import product


class LoggedMetrics:
    # metric,                 str: "loss", "acc", "precision", "recall", "f1"
    # phases,           List[str]: subset of ["train", "val", "test"]
    # levels,           List[str]: subset of ["token", "entity"]
    # tag_groups,       List[str]: subset of ["all", "fil", "ind"]
    # averaging_groups: List[str]: subset of ["simple", "micro", "macro"]
    logged_metrics = [
        # GENERAL
        ("loss", ["train", "val", "test"], ["all"], ["token"], ["simple"]),
        ("acc", ["train", "val", "test"], ["all"], ["token"], ["simple"]),
        # VAL
        ("f1", ["val"], ["all"], ["token"], ["micro"]),
        ("f1", ["val"], ["fil"], ["token", "entity"], ["micro"]),
        # TEST
        ("precision", ["test"], ["all"], ["token"], ["micro", "macro"]),
        ("precision", ["test"], ["fil"], ["token", "entity"], ["micro", "macro"]),
        ("precision", ["test"], ["ind"], ["token", "entity"], ["micro"]),
        ("precision", ["test"], ["O"], ["token"], ["micro"]),
        ("recall", ["test"], ["all"], ["token"], ["micro", "macro"]),
        ("recall", ["test"], ["fil"], ["token", "entity"], ["micro", "macro"]),
        ("recall", ["test"], ["ind"], ["token", "entity"], ["micro"]),
        ("recall", ["test"], ["O"], ["token"], ["micro"]),
        ("f1", ["test"], ["all"], ["token"], ["micro", "macro"]),
        ("f1", ["test"], ["fil"], ["token", "entity"], ["micro", "macro"]),
        ("f1", ["test"], ["ind"], ["token", "entity"], ["micro"]),
        ("f1", ["test"], ["O"], ["token"], ["micro"]),
        (
            "numberofclasses",
            ["test"],
            ["fil"],
            ["token", "entity"],
            ["macro"],
        ),  # special
    ]

    @classmethod
    def as_flat_list(cls):
        """
        return logged_metrics as flat list
        ----------------------------------
        :return: [list] of [str], e.g. ['token_all_precision_micro', 'token_fil_precision_macro', ..]
        """
        flat_list = list()
        for metric, _, tag_groups, levels, averaging_groups in cls.logged_metrics:
            for tag_group, level in product(tag_groups, levels):
                if averaging_groups == ["simple"]:
                    flat_list.append(f"{level}_{tag_group}_{metric}")
                else:
                    for averaging_group in averaging_groups:
                        flat_list.append(
                            f"{level}_{tag_group}_{metric}_{averaging_group}"
                        )
        return flat_list

    def __init__(self):
        for metric, phases, tag_groups, levels, averaging_groups in self.logged_metrics:
            assert isinstance(metric, str)
            assert isinstance(phases, list)
            assert isinstance(tag_groups, list)
            assert isinstance(levels, list)
            assert isinstance(averaging_groups, list)

    def get_metrics(
        self,
        required_tag_groups: List[str] = None,
        required_phases: List[str] = None,
        required_levels: List[str] = None,
        required_averaging_groups: List[str] = None,
        exclude: List[str] = None,
    ) -> List[str]:
        """
        get metrics, filtered
        ---------------------
        :param required_tag_groups:       [list] of tag_groups that are required, e.g. ['all']
        :param required_phases:           [list] of phases that are required, e.g. ['train']
        :param required_levels:           [list] of levels that are required, e.g. ['token']
        :param required_averaging_groups: [list] averaging groups that are required, e.g. ['micro', 'macro']
        :param exclude:                   [list] of metrics to exclude, e.g. ['loss']
        :return: filtered_metrics:        [list] of [str], e.g. ['precision', 'recall']
        """
        filtered_metrics = list()
        for metric, phases, tag_groups, levels, averaging_groups in self.logged_metrics:
            add_metric_1 = (
                True
                if required_tag_groups is None
                else all(
                    [
                        required_tag_group in tag_groups
                        for required_tag_group in required_tag_groups
                    ]
                )
            )
            add_metric_2 = (
                True
                if required_phases is None
                else all(
                    [required_phase in phases for required_phase in required_phases]
                )
            )
            add_metric_3 = (
                True
                if required_levels is None
                else all(
                    [required_level in levels for required_level in required_levels]
                )
            )
            add_metric_4 = (
                True
                if required_averaging_groups is None
                else all(
                    [
                        required_averaging_group in averaging_groups
                        for required_averaging_group in required_averaging_groups
                    ]
                )
            )
            add_metric_5 = True if exclude is None else metric not in exclude

            if (
                add_metric_1
                and add_metric_2
                and add_metric_3
                and add_metric_4
                and add_metric_5
            ):
                filtered_metrics.append(metric)

        return list(set(filtered_metrics))
