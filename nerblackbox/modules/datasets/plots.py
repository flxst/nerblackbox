import matplotlib.pylab as plt


class Plots:

    clr = ["k", "g", "orange", "r"]
    shift = [-0.3, -0.1, 0.1, 0.3]
    width = 0.1
    column_names = ["tags/sentence", "tags relative w/ 0", "tags relative w/o 0"]

    def __init__(self, _stats_aggregated, _num_sentences):
        self.stats_aggregated = _stats_aggregated
        self.num_sentences = _num_sentences
        self.num_tokens = None
        self.tags = list(self.stats_aggregated["total"].index)

    def plot(self, fig_path=None):
        ################################################################################################################
        # 3. START
        ################################################################################################################
        phases = ["train", "val", "test"]
        phases_all = ["total"] + phases

        self.num_tokens = {
            phase: self.get_tokens(self.stats_aggregated[phase]) for phase in phases_all
        }
        columns = {
            column_name: {
                phase: self.get_columns(self.stats_aggregated[phase], column_name)
                for phase in phases_all
            }
            for column_name in self.column_names
        }

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax0, ax1, ax2, ax3 = ax.flatten()
        self.plot_sentences_and_tokens(ax0, self.num_sentences, self.num_tokens)
        self.plot_column(
            ax1,
            columns,
            "tags/sentence",
            legend={"loc": "upper right", "bbox_to_anchor": (1.24, 1.0)},
        )
        self.plot_column(ax2, columns, "tags relative w/ 0", y_normalize=True)
        self.plot_column(ax3, columns, "tags relative w/o 0", y_normalize=True)

        if fig_path is not None:
            print("\nsave figure at", fig_path)
            plt.savefig(fig_path)

    ####################################################################################################################
    # GET DATA FROM DF
    ####################################################################################################################
    @staticmethod
    def get_tokens(df):
        return df.loc[:, "tags"].sum()

    @staticmethod
    def get_columns(df, column_name: str):
        return df.loc[:, column_name].to_dict()

    ####################################################################################################################
    # PLOT SENTENCES AND TOKENS
    ####################################################################################################################
    def plot_sentences_and_tokens(self, _ax, _sentences, _tokens, legend=False):
        tags = ["sentences", "tokens"]
        xs = list(range(len(tags)))
        labels = list(_sentences.keys())

        def normalize(_list):
            return [elem / _list[0] for elem in _list], int(_list[0])

        values_sentences, total_sentences = normalize(list(_sentences.values()))
        values_tokens, total_tokens = normalize(list(_tokens.values()))

        for n, tag in enumerate(tags):
            x = xs[n]
            for i in range(len(self.shift)):
                if n == 0:
                    _ax.bar(
                        x + self.shift[i],
                        values_sentences[i],
                        label=labels[i] if n == 0 else None,
                        width=self.width,
                        color=self.clr[i],
                    )
                else:
                    _ax.bar(
                        x + self.shift[i],
                        values_tokens[i],
                        label=labels[i] if n == 0 else None,
                        width=self.width,
                        color=self.clr[i],
                    )
            if legend:
                _ax.legend()
            _ax.set_xticks(xs)
            _ax.set_xticklabels(tags)
            _ax.set_ylim([0, 1])
            _ax.set_yticks([1])
            _ax.set_yticklabels([total_sentences])

            _ax2 = _ax.twinx()
            _ax2.set_ylim([0, 1])
            _ax2.set_yticks([1])
            _ax2.set_yticklabels([total_tokens])

    def plot_column(self, _ax, _columns, column_name, y_normalize=False, legend=None):
        column = _columns[column_name]

        xs = list(range(len(self.tags)))
        labels = list(column.keys())

        for n, tag in enumerate(self.tags):
            x = xs[n]
            for i in range(len(self.shift)):
                _ax.bar(
                    x + self.shift[i],
                    float(column[labels[i]][tag]),
                    label=labels[i] if n == 0 else None,
                    width=self.width,
                    color=self.clr[i],
                )
            if n == len(self.tags) - 1:
                if legend is not None:
                    _ax.legend(**legend)
                _ax.set_xticks(xs)
                _ax.set_xticklabels(self.tags)
        _ax.set_ylabel(column_name)
        if y_normalize:
            _ax.set_ylim([0, 1])
