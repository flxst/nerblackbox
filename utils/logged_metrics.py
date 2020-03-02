
class LoggedMetrics:

    def __init__(self, logged_metrics):
        """
        :param logged_metrics: [list] of [tuples] w/ (metric, tags_list, micro_macro_list),
                                                  e.g. [('precision', ['all', 'fil'], ['micro', 'macro']), ..]
        """
        for metric, tags_list, micro_macro_list in logged_metrics:
            assert isinstance(metric, str)
            assert isinstance(tags_list, list)
            assert isinstance(micro_macro_list, list)

        self.logged_metrics = logged_metrics

    def get_metrics(self, tags: list = None, micro_macros: list = None, exclude: list = None):
        """
        get metrics, filtered
        ---------------------
        :param tags:         [list] of tags that are required, e.g. ['all']
        :param micro_macros: [list] simple/micro/macro that is required, e.g. ['micro', 'macro']
        :param exclude:      [list] of metrics to exclude, e.g. ['loss']
        :return:
        """
        filtered_metrics = list()
        for metric, tags_list, micro_macro_list in self.logged_metrics:
            add_metric_1 = \
                True if tags is None else all([tag in tags_list for tag in tags])
            add_metric_2 = \
                True if micro_macros is None else all([micro_macro in micro_macro_list for micro_macro in micro_macros])
            add_metric_3 = \
                True if exclude is None else metric not in exclude

            if add_metric_1 and add_metric_2 and add_metric_3:
                filtered_metrics.append(metric)

        return filtered_metrics

    def as_flat_list(self):
        """
        return logged_metrics as flat list
        ----------------------------------
        :return: [list] of [str], e.g. ['all_precision_micro', 'all_precision_macro', 'fil_precision_micro', ..]
        """
        flat_list = list()
        for metric, tags_list, micro_macro_list in self.logged_metrics:
            for tags in tags_list:
                if micro_macro_list == ['simple']:
                    flat_list.append(f'{tags}_{metric}')
                else:
                    for style in micro_macro_list:
                        flat_list.append(f'{tags}_{metric}_{style}')
        return flat_list
