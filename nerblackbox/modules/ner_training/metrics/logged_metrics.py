class LoggedMetrics:
    def __init__(self, logged_metrics=None):
        """
        :param logged_metrics: [list] of [tuples] w/ (metric, phase_list, tags_list, micro_macro_list),
                                                  e.g. [('precision', ['val'], ['all', 'fil'], ['micro', 'macro']),..]
        """
        if logged_metrics is None:
            self.logged_metrics = [
                ("loss", ["train", "val", "test"], ["all+"], ["simple"]),
                ("acc", ["train", "val", "test"], ["all+"], ["simple"]),
                (
                    "precision",
                    ["val", "test"],
                    ["all+", "all", "fil"],
                    ["micro", "macro"],
                ),
                ("precision", ["val", "test"], ["chk", "ind"], ["micro"]),
                ("recall", ["val", "test"], ["all+", "all", "fil"], ["micro", "macro"]),
                ("recall", ["val", "test"], ["chk", "ind"], ["micro"]),
                ("f1", ["val", "test"], ["all+", "all", "fil"], ["micro", "macro"]),
                ("f1", ["val", "test"], ["chk", "ind"], ["micro"]),
            ]
        else:
            self.logged_metrics = logged_metrics
        for metric, phase_list, tags_list, micro_macro_list in self.logged_metrics:
            assert isinstance(metric, str)
            assert isinstance(phase_list, list)
            assert isinstance(tags_list, list)
            assert isinstance(micro_macro_list, list)

    def get_metrics(
        self,
        tag_group: list = None,
        phase_group: list = None,
        micro_macro_group: list = None,
        exclude: list = None,
    ):
        """
        get metrics, filtered
        ---------------------
        :param tag_group:          [list] of tags that are required, e.g. ['all']
        :param phase_group:        [list] of phases that are required, e.g. ['train']
        :param micro_macro_group:  [list] simple/micro/macro that is required, e.g. ['micro', 'macro']
        :param exclude:            [list] of metrics to exclude, e.g. ['loss']
        :return: filtered_metrics: [list] of [str], e.g. ['precision', 'recall']
        """
        filtered_metrics = list()
        for metric, phase_list, tags_list, micro_macro_list in self.logged_metrics:
            add_metric_1 = (
                True
                if tag_group is None
                else all([tag in tags_list for tag in tag_group])
            )
            add_metric_2 = (
                True
                if phase_group is None
                else all([phase in phase_list for phase in phase_group])
            )
            add_metric_3 = (
                True
                if micro_macro_group is None
                else all(
                    [
                        micro_macro in micro_macro_list
                        for micro_macro in micro_macro_group
                    ]
                )
            )
            add_metric_4 = True if exclude is None else metric not in exclude

            if add_metric_1 and add_metric_2 and add_metric_3 and add_metric_4:
                filtered_metrics.append(metric)

        return list(set(filtered_metrics))

    def as_flat_list(self):
        """
        return logged_metrics as flat list
        ----------------------------------
        :return: [list] of [str], e.g. ['all_precision_micro', 'all_precision_macro', 'fil_precision_micro', ..]
        """
        flat_list = list()
        for metric, _, tags_list, micro_macro_list in self.logged_metrics:
            for tags in tags_list:
                if micro_macro_list == ["simple"]:
                    flat_list.append(f"{tags}_{metric}")
                else:
                    for style in micro_macro_list:
                        flat_list.append(f"{tags}_{metric}_{style}")
        return flat_list
