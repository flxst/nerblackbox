import os
import subprocess
import json
import pandas as pd
from abc import ABC, abstractmethod

from os.path import join

from nerblackbox.modules.utils.util_functions import get_dataset_path
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.datasets.plots import Plots
from nerblackbox.modules.datasets.formatter.util_functions import get_ner_tag_mapping
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger


class BaseFormatter(ABC):
    def __init__(self, ner_dataset, ner_tag_list):
        """
        :param ner_dataset:  [str] 'swedish_ner_corpus' or 'suc'
        :param ner_tag_list: [list] of [str], e.g. ['PER', 'LOC', ..]
        """
        self.ner_dataset = ner_dataset
        self.ner_tag_list = ner_tag_list
        self.dataset_path = get_dataset_path(ner_dataset)
        self.stats_aggregated = None
        self.num_tokens = None
        self.num_sentences = None

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def get_data(self, verbose: bool):
        """
        I: get data
        -----------
        :param verbose: [bool]
        :return: -
        """
        pass

    @abstractmethod
    def create_ner_tag_mapping(self):
        """
        II: customize ner_training tag mapping if wanted
        -------------------------------------
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        pass

    @abstractmethod
    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        pass

    @abstractmethod
    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float]
        :return: -
        """
        pass

    ####################################################################################################################
    # BASE METHODS
    ####################################################################################################################
    def create_directory(self):
        """
        0: create directory for dataset
        -------------------------------
        :return: -
        """
        directory_path = (
            f'{env_variable("DIR_DATASETS")}/{self.ner_dataset}/analyze_data'
        )
        os.makedirs(directory_path, exist_ok=True)

        bash_cmd = (
            f'echo "*" > {env_variable("DIR_DATASETS")}/{self.ner_dataset}/.gitignore'
        )
        try:
            subprocess.run(bash_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)

    def create_ner_tag_mapping_json(self, modify: bool):
        """
        II: create customized ner_training tag mapping to map tags in original data to tags in formatted data
        ----------------------------------------------------------------------------------------------
        :param modify:      [bool], if True: modify tags as specified in method modify_ner_tag_mapping()
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        if modify:
            ner_tag_mapping = self.create_ner_tag_mapping()
        else:
            ner_tag_mapping = dict()

        json_path = (
            f'{env_variable("DIR_DATASETS")}/{self.ner_dataset}/ner_tag_mapping.json'
        )
        with open(json_path, "w") as f:
            json.dump(ner_tag_mapping, f)

        print(f"> dumped the following dict to {json_path}:")
        print(ner_tag_mapping)

    ####################################################################################################################
    # HELPER: WRITE FORMATTED
    ####################################################################################################################
    def _write_formatted_csv(self, phase, rows):
        """
        III: format data
        ----------------------------------------------
        :param phase:         [str] 'train' or 'test'
        :param rows:          [list] of [list] of [str], e.g. [['Inger', 'PER'], ['säger', '0'], ..]
        :return: -
        """
        file_path = join(self.dataset_path, f"{phase}_formatted.csv")

        # ner tag mapping
        ner_tag_mapping = get_ner_tag_mapping(
            path=join(self.dataset_path, "ner_tag_mapping.json")
        )

        # processing
        data = list()
        tags = list()
        sentence = list()
        for row in rows:
            if len(row) == 2:
                sentence.append(row[0])
                tags.append(
                    ner_tag_mapping(row[1]) if row[1] != "0" else "O"
                )  # replace zeros by capital O (!)
            else:
                if len(row) != 0:
                    print(
                        f"ATTENTION!! row with length = {len(row)} found (should be 0 or 2): {row}"
                    )
                if len(tags) and len(sentence):
                    data.append([" ".join(tags), " ".join(sentence)])
                    tags = list()
                    sentence = list()

        df = pd.DataFrame(data)
        df.to_csv(file_path, sep="\t", header=None, index=None)
        print(
            f"> phase = {phase}: wrote {len(rows)} words in {len(df)} sentences to {file_path}"
        )

    ####################################################################################################################
    # HELPER: READ FORMATTED
    ####################################################################################################################
    def _read_formatted_csvs(self, phases):
        """
        IV: resplit data
        ----------------
        :param phases: [list] of [str] to read formatted csvs from, e.g. ['val', 'test']
        :return: [pd DataFrame]
        """
        df_phases = [self._read_formatted_csv(phase) for phase in phases]
        return pd.concat(df_phases, ignore_index=True)

    def _read_formatted_csv(self, phase):
        """
        IV: resplit data
        ----------------
        :param phase: [str] csvs to read formatted df from, e.g. 'val'
        :return: df: [pd DataFrame]
        """
        formatted_file_path = join(self.dataset_path, f"{phase}_formatted.csv")
        try:
            df = pd.read_csv(formatted_file_path, sep="\t", header=None)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return df

    ####################################################################################################################
    # HELPER: WRITE FINAL
    ####################################################################################################################
    @staticmethod
    def _split_off_validation_set(_df_original, _val_fraction):
        """
        IV: resplit data
        ----------------
        :param _df_original:    [pd DataFrame]
        :param _val_fraction:   [float] between 0 and 1
        :return: _df_new:       [pd DataFrame]
        :return: _df_val:       [pd DataFrame]
        """
        split_index = int(len(_df_original) * (1 - _val_fraction))
        _df_new = _df_original.iloc[:split_index]
        _df_val = _df_original.iloc[split_index:]
        return _df_new, _df_val

    def _write_final_csv(self, phase, df):
        """
        IV: resplit data
        ----------------
        :param phase: [str], 'train', 'val' or 'test'
        :param df: [pd DataFrame]
        :return:
        """
        file_path = join(self.dataset_path, f"{phase}.csv")
        df.to_csv(file_path, sep="\t", index=False, header=None)

    ####################################################################################################################
    def read_formatted_csv(self, phase):
        """
        V: read formatted csv files
        ----------------------------------------------
        :param phase:         [str] 'train' or 'test'
        :return: num_sentences:    [int]
                 stats_aggregated: [pandas Series] with indices = tags, values = number of occurrences
        """
        file_path = join(self.dataset_path, f"{phase}.csv")

        columns = ["O"] + self.ner_tag_list

        try:
            df = pd.read_csv(file_path, sep="\t")
        except pd.io.common.EmptyDataError:
            df = None

        stats = pd.DataFrame([], columns=columns)

        if df is not None:
            tags = (
                df.iloc[:, 0]
                .apply(lambda x: x.split())
                .apply(lambda x: [elem.split("-")[-1] for elem in x])
            )

            for column in columns:
                stats[column] = tags.apply(
                    lambda x: len([elem for elem in x if elem == column])
                )

            assert len(df) == len(stats)

        num_sentences = len(stats)
        stats_aggregated = stats.sum().to_frame().astype(int)
        stats_aggregated.columns = ["tags"]

        return num_sentences, stats_aggregated

    @staticmethod
    def get_tokens(df):
        return df.loc[:, "tags"].sum()

    def analyze_data(self):
        """
        V: analyze data
        ----------------
        :created attr: stats_aggregated: [dict] w/ keys = 'total', 'train', 'val', 'test' & values = [df]
        :return: -
        """
        log_file = join(self.dataset_path, "analyze_data", f"{self.ner_dataset}.log")
        default_logger = DefaultLogger(
            __file__, log_file=log_file, level="info", mode="w"
        )

        self.num_tokens = {"total": 0}
        self.num_sentences = {"total": 0}
        self.stats_aggregated = {"total": None}

        phases = ["train", "val", "test"]
        phases_all = ["total"] + phases

        for phase in phases:
            (
                self.num_sentences[phase],
                _stats_aggregated_phase,
            ) = self.read_formatted_csv(phase)
            self.num_sentences["total"] += self.num_sentences[phase]
            if self.stats_aggregated["total"] is None:
                self.stats_aggregated["total"] = _stats_aggregated_phase
            else:
                self.stats_aggregated["total"] = (
                    self.stats_aggregated["total"] + _stats_aggregated_phase
                )

            self.stats_aggregated[phase] = self._stats_aggregated_add_columns(
                _stats_aggregated_phase, self.num_sentences[phase]
            )

        num_sentences_total = self.num_sentences["total"]
        self.stats_aggregated["total"] = self._stats_aggregated_add_columns(
            self.stats_aggregated["total"], self.num_sentences["total"]
        )
        self.num_tokens = {
            phase: self.get_tokens(self.stats_aggregated[phase]) for phase in phases_all
        }
        num_tokens_total = self.num_tokens["total"]

        # print/log
        for phase in phases:
            default_logger.log_info("")
            default_logger.log_info(f">>> {phase} <<<<")
            default_logger.log_info(
                f"num_sentences = {self.num_sentences[phase]} "
                f"({100*self.num_sentences[phase]/num_sentences_total:.2f}% of total = {num_sentences_total})"
            )
            default_logger.log_info(
                f"num_tokens = {self.num_tokens[phase]} "
                f"({100*self.num_tokens[phase]/num_tokens_total:.2f}% of total = {num_tokens_total})"
            )
            default_logger.log_info(self.stats_aggregated[phase])

        default_logger.log_info("")
        default_logger.log_info(f"num_sentences = {self.num_sentences}")
        default_logger.log_info(f"num_tokens = {self.num_tokens}")
        default_logger.log_info(f">>> total <<<<")
        default_logger.log_info(self.stats_aggregated["total"])

    def plot_data(self):
        fig_path = join(self.dataset_path, "analyze_data", f"{self.ner_dataset}.png")
        Plots(self.stats_aggregated, self.num_sentences).plot(fig_path=fig_path)

    @staticmethod
    def _stats_aggregated_add_columns(df, number_of_sentences):
        """
        V: analyze data
        ----------------
        :param df: ..
        :param number_of_sentences: ..
        :return: ..
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

        return df
