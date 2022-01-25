from os.path import join, isfile
from typing import List, Dict, Optional, Tuple
import pandas as pd

from nerblackbox.modules.datasets.plots import Plots
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger


class Analyzer:

    phases = ["train", "val", "test"]
    phases_all = ["total"] + phases

    def __init__(self, ner_dataset: str, ner_tag_list: List[str], dataset_path: str):
        """
        Args:
            ner_dataset:  'swedish_ner_corpus' or 'suc'
            ner_tag_list: e.g. ['PER', 'LOC', ..]
            dataset_path: e.g. DATA_DIR/datasets/swedish_ner_corpus
        """
        self.ner_dataset = ner_dataset
        self.ner_tag_list = ner_tag_list
        self.dataset_path = dataset_path
        self.stats_aggregated: Dict[str, Optional[pd.DataFrame]] = {"total": None}
        self.num_tokens: Dict[str, int] = {"total": 0}
        self.num_sentences: Dict[str, int] = {"total": 0}
        self.analysis_flag: bool = True

    def analyze_data(self, write_log: bool = True) -> None:
        """
        V: analyze data

        Args:
            write_log: whether to write log file (should always be True except for testing)

        Created Attr:
            stats_aggregated: [dict] w/ keys = 'total', 'train', 'val', 'test' & values = [df]
            num_tokens:       [dict] w/ keys = 'total', 'train', 'val', 'test' & values = [int]
            num_sentences:    [dict] w/ keys = 'total', 'train', 'val', 'test' & values = [int]

        Returns: -
        """
        for phase in self.phases:
            file_path = join(self.dataset_path, f"{phase}.csv")
            if not isfile(file_path):
                file_path_jsonl = join(self.dataset_path, f"{phase}.jsonl")
                if isfile(file_path_jsonl):
                    self.analysis_flag = False
                    return None
                else:
                    raise Exception(f"ERROR! file = {file_path} not found!")

            (
                self.num_sentences[phase],  # int, e.g. 4
                _stats_aggregated_phase,  # pd.DataFrame w/ col "tags", index = tags, values = tag occurrences
            ) = self._read_final_csv(phase)

            self.num_sentences["total"] += self.num_sentences[phase]
            if self.stats_aggregated["total"] is None:
                self.stats_aggregated["total"] = _stats_aggregated_phase
            else:
                self.stats_aggregated["total"] = (
                    self.stats_aggregated["total"] + _stats_aggregated_phase
                )

            self.stats_aggregated[phase] = self._stats_aggregated_extend(
                _stats_aggregated_phase, self.num_sentences[phase]
            )

        num_sentences_total = self.num_sentences["total"]
        self.stats_aggregated["total"] = self._stats_aggregated_extend(
            self.stats_aggregated["total"], self.num_sentences["total"]
        )

        # num_tokens
        self.num_tokens = {
            phase: self._get_num_tokens(self.stats_aggregated[phase])
            for phase in self.phases_all
        }
        num_tokens_total = self.num_tokens["total"]

        if write_log:  # pragma: no cover
            self._write_log_file(num_sentences_total, num_tokens_total)

    def plot_data(self) -> None:  # pragma: no cover
        if self.analysis_flag:
            fig_path = join(
                self.dataset_path, "analyze_data", f"{self.ner_dataset}.png"
            )
            Plots(self.stats_aggregated, self.num_sentences).plot(fig_path=fig_path)

    ####################################################################################################################
    # HELPER METHODS: analyze_data #####################################################################################
    ####################################################################################################################
    def _read_final_csv(self, phase: str) -> Tuple[int, pd.DataFrame]:
        """
        V: read final csv files

        Args:
            phase:         [str] 'train' or 'test'

        Returns:
            num_sentences:    [int]
            stats_aggregated: [pandas DataFrame] with indices = tags, column = "tag", values = number of tag occurrences
        """
        file_path = join(self.dataset_path, f"{phase}.csv")
        assert isfile(file_path), f"ERROR! file = {file_path} not found!"

        columns = ["O"] + self.ner_tag_list

        df = pd.read_csv(file_path, sep="\t", header=None, names=["labels", "text"])

        stats = pd.DataFrame([], columns=columns)

        if df is not None:
            tags = (
                df.iloc[:, 0]
                .apply(lambda x: str(x).split())
                .apply(lambda x: [elem.split("-")[-1] for elem in x])
            )

            for column in columns:
                stats[column] = tags.apply(
                    lambda x: len([elem for elem in x if elem == column])
                )

            assert len(df) == len(
                stats
            ), f"ERROR! len(df) = {len(df)} is not equal to len(stats) = {len(stats)}"

        num_sentences = len(stats)
        stats_aggregated = stats.sum().to_frame().astype(int)
        stats_aggregated.columns = ["tags"]

        return num_sentences, stats_aggregated

    @staticmethod
    def _stats_aggregated_extend(
        df: pd.DataFrame, number_of_sentences: int
    ) -> pd.DataFrame:
        """
        V: analyze data
        extend dataframe by adding columns

        Args:
            df: with column "tags"
            number_of_sentences: e.g. 1000

        Returns:
            df_extended: with columns ["tags", "tags/sentence", "tags relative w/ 0", "tags relative w/o 0"]
        """
        df["tags/sentence"] = df["tags"] / float(number_of_sentences)
        df["tags/sentence"] = df["tags/sentence"].apply(lambda x: "{:.2f}".format(x))

        # relative tags w/ 0
        number_of_occurrences = df["tags"].sum()
        df["tags relative w/ 0"] = df["tags"] / number_of_occurrences
        df["tags relative w/ 0"] = df["tags relative w/ 0"].apply(
            lambda x: "{:.2f}".format(x)
        )

        # relative tags w/o 0
        number_of_filtered_occurrences = df["tags"].sum() - df.loc["O"]["tags"]
        df["tags relative w/o 0"] = df["tags"] / number_of_filtered_occurrences
        df["tags relative w/o 0"]["O"] = 0
        df["tags relative w/o 0"] = df["tags relative w/o 0"].apply(
            lambda x: "{:.2f}".format(x)
        )

        df = df.reindex(
            ["tags", "tags/sentence", "tags relative w/ 0", "tags relative w/o 0"],
            axis=1,
        )

        return df

    @staticmethod
    def _get_num_tokens(df: pd.DataFrame) -> int:
        """
        Args:
            df: DataFrame with column "tags"

        Returns:
            num_tokens: sum of all values in column "tags"
        """
        assert (
            "tags" in df.columns
        ), f"ERROR! df.columns = {df.columns} does not contain 'tags'"
        return df.loc[:, "tags"].sum()

    def _write_log_file(
        self, num_sentences_total: int, num_tokens_total: int
    ) -> None:  # pragma: no cover
        # print/log
        log_file = join(self.dataset_path, "analyze_data", f"{self.ner_dataset}.log")
        default_logger = DefaultLogger(
            __file__, log_file=log_file, level="info", mode="w"
        )
        for phase in self.phases:
            default_logger.log_info("")
            default_logger.log_info(f">>> {phase} <<<<")
            default_logger.log_info(
                f"num_sentences = {self.num_sentences[phase]} "
                f"({100 * self.num_sentences[phase] / num_sentences_total:.2f}% of total = {num_sentences_total})"
            )
            default_logger.log_info(
                f"num_tokens = {self.num_tokens[phase]} "
                f"({100 * self.num_tokens[phase] / num_tokens_total:.2f}% of total = {num_tokens_total})"
            )
            default_logger.log_info(self.stats_aggregated[phase])

        default_logger.log_info("")
        default_logger.log_info(f"num_sentences = {self.num_sentences}")
        default_logger.log_info(f"num_tokens = {self.num_tokens}")
        default_logger.log_info(f">>> total <<<<")
        default_logger.log_info(self.stats_aggregated["total"])
