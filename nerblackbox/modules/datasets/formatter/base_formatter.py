import os
from os.path import join, isfile
import subprocess
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union, Any
import random

from nerblackbox.modules.utils.util_functions import get_dataset_path
from nerblackbox.modules.utils.env_variable import env_variable
from nerblackbox.modules.datasets.formatter.util_functions import get_ner_tag_mapping
from nerblackbox.modules.datasets.analyzer import Analyzer

SEED_SHUFFLE = {
    "train": 4,
    "val": 5,
    "test": 6,
}

SENTENCES_ROWS_PRETOKENIZED = List[List[List[str]]]
SENTENCES_ROWS_UNPRETOKENIZED = List[Dict[str, Any]]
SENTENCES_ROWS = Union[SENTENCES_ROWS_PRETOKENIZED, SENTENCES_ROWS_UNPRETOKENIZED]


class BaseFormatter(ABC):
    def __init__(self, ner_dataset: str, ner_tag_list: List[str]):
        """
        Args:
            ner_dataset:  'swedish_ner_corpus' or 'suc'
            ner_tag_list: e.g. ['PER', 'LOC', ..]
        """
        self.ner_dataset: str = ner_dataset
        self.ner_tag_list: List[str] = ner_tag_list
        self.dataset_path: str = get_dataset_path(ner_dataset)
        self.file_name: Dict[str, str] = {}
        self.analyzer = Analyzer(self.ner_dataset, self.ner_tag_list, self.dataset_path)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def get_data(self, verbose: bool) -> None:  # pragma: no cover
        """
        I: get data

        Args:
            verbose: [bool]
        """
        pass

    @abstractmethod
    def create_ner_tag_mapping(self) -> Dict[str, str]:  # pragma: no cover
        """
        II: customize ner_training tag mapping if wanted

        Returns:
            ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        pass

    @abstractmethod
    def format_data(
        self, shuffle: bool = True, write_csv: bool = True
    ) -> Optional[SENTENCES_ROWS]:  # pragma: no cover
        """
        III: format data

        Args:
            shuffle: whether to shuffle rows of dataset
            write_csv: whether to write dataset to csv (should always be True except for testing)
        """
        pass

    def set_original_file_paths(self) -> None:  # pragma: no cover
        """
        III: format data

        Changed Attributes:
            file_paths: [Dict[str, str]], e.g. {'train': <path_to_train_csv>, 'val': ..}

        Returns: -
        """
        pass

    @abstractmethod
    def _parse_row(self, _row: str) -> List[str]:  # pragma: no cover
        """
        III: format data

        Args:
            _row: e.g. "Det PER X B"

        Returns:
            _row_list: e.g. ["Det", "PER", "X", "B"]
        """
        pass

    def _format_original_file(
        self, _row_list: List[str]
    ) -> Optional[List[str]]:  # pragma: no cover
        """
        III: format data

        Args:
            _row_list: e.g. ["test", "PER", "X", "B"]

        Returns:
            _row_list_formatted: e.g. ["test", "B-PER"]
        """
        pass

    @abstractmethod
    def resplit_data(
        self, val_fraction: float, write_csv: bool
    ) -> Optional[Tuple[pd.DataFrame, ...]]:  # pragma: no cover
        """
        IV: resplit data

        Args:
            val_fraction: [float], e.g. 0.3
            write_csv: whether to write dataset to csv (should always be True except for testing)
        """
        pass

    ####################################################################################################################
    # BASE METHODS
    ####################################################################################################################
    def create_directory(self) -> None:  # pragma: no cover
        """
        0: create directory for dataset
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

    def create_ner_tag_mapping_json(self, modify: bool) -> None:  # pragma: no cover
        """
        II: create customized ner_training tag mapping to map tags in original data to tags in formatted data

        Args:
            modify:      [bool], if True: modify tags as specified in method modify_ner_tag_mapping()

        Returns: -
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
    # HELPER: READ ORIGINAL
    ####################################################################################################################
    def _read_original_file(self, phase: str) -> SENTENCES_ROWS_PRETOKENIZED:
        """
        III: format data

        Args:
            phase: 'train', 'val', 'test'

        Returns:
            sentences_rows: e.g. (-pretokenized-)
                            [
                                [['Inger', 'PER'], ['säger', '0'], .., []],
                                [['Det', '0'], .., []]
                            ]
        """
        self.set_original_file_paths()
        file_path_original = join(self.dataset_path, self.file_name[phase])

        _sentences_rows = list()
        if isfile(file_path_original):
            _sentence = list()
            with open(file_path_original) as f:
                for row in f.readlines():
                    row_list = self._parse_row(row)
                    if len(row_list) > 0:
                        row_list_formatted = self._format_original_file(row_list)
                        if row_list_formatted is not None:
                            _sentence.append(row_list_formatted)
                    else:
                        if len(_sentence):
                            _sentences_rows.append(_sentence)
                        _sentence = list()
            print(f"\n> read {file_path_original}")
        else:  # pragma: no cover
            raise Exception(f"ERROR! could not find file {file_path_original}!")

        return _sentences_rows

    ####################################################################################################################
    # HELPER: WRITE FORMATTED
    ####################################################################################################################
    def _write_formatted_csv(
        self, phase: str, sentences_rows: SENTENCES_ROWS_PRETOKENIZED
    ) -> None:  # pragma: no cover
        """
        III: format data

        Args:
            phase: 'train', 'val', 'test'
            sentences_rows: e.g. (-pretokenized-)
                            [
                                [['Inger', 'PER'], ['säger', '0'], .., []],
                                [['Det', '0'], .., []]
                            ]

        Returns: -
        """
        sentences_rows_formatted = self._format_sentences_rows(sentences_rows)

        df = pd.DataFrame(sentences_rows_formatted)
        file_path = join(self.dataset_path, f"{phase}_formatted.csv")
        df.to_csv(file_path, sep="\t", header=False, index=False)
        print(f"> phase = {phase}: wrote {len(df)} sentences to {file_path}")

    def _write_formatted_jsonl(
        self, phase: str, sentences_rows: SENTENCES_ROWS_UNPRETOKENIZED
    ) -> None:  # pragma: no cover
        """
        save to jsonl file

        Args:
            phase: 'train', 'val', 'test'
            sentences_rows: e.g. (-unpretokenized-)
                            [
                                {
                                    'text': 'Inger säger ..',
                                    'tags': [{'token': 'Inger', 'tag': 'PER', 'char_start': 0, 'char_end': 5}, ..],
                                },
                                {
                                    'text': 'Det ..',
                                    'tags': [{..}, ..]
                                }
                            ]

        Returns: -
        """
        file_path = join(self.dataset_path, f"{phase}_formatted.jsonl")
        with open(file_path, "w") as file:
            for sentence_row in sentences_rows:
                file.write(json.dumps(sentence_row, ensure_ascii=False) + "\n")
        print(
            f"> phase = {phase}: wrote {len(sentences_rows)} sentences to {file_path}"
        )

    def _format_sentences_rows(
        self, sentences_rows: SENTENCES_ROWS_PRETOKENIZED
    ) -> List[Tuple[str, str]]:
        """
        III: format data

        Args:
            sentences_rows: e.g. (-pretokenized-)
                            [
                                [['Inger', 'PER'], ['säger', '0'], .., []],
                                [['Det', '0'], .., []]
                            ]

        Returns:
            sentences_rows_formatted, e.g. (-pretokenized-)
                            [
                                ('PER O', 'Inger säger'),
                                ('O', 'Det'),
                            ]
        """

        # ner tag mapping
        ner_tag_mapping = get_ner_tag_mapping(
            path=join(self.dataset_path, "ner_tag_mapping.json")
        )

        # processing
        sentences_rows_formatted = list()
        for sentence in sentences_rows:
            text_list = list()
            tags_list = list()
            for row in sentence:
                assert (
                    len(row) == 2
                ), f"ERROR! row with length = {len(row)} found (should be 2): {row}"
                text_list.append(row[0])
                tags_list.append(
                    ner_tag_mapping(row[1]) if row[1] != "0" else "O"
                )  # replace zeros by capital O (!)
            sentences_rows_formatted.append((" ".join(tags_list), " ".join(text_list)))

        return sentences_rows_formatted

    @staticmethod
    def _convert_iob1_to_iob2(
        sentences_rows_iob1: SENTENCES_ROWS_PRETOKENIZED,
    ) -> SENTENCES_ROWS_PRETOKENIZED:
        """
        III: format data
        convert tags from IOB1 to IOB2 format

        Args:
            sentences_rows_iob1: e.g. (-pretokenized-)
                            [
                                [['Inger', 'I-PER'], ['säger', '0'], .., []],
                            ]

        Returns:
            sentences_rows_iob2: e.g. (-pretokenized-)
                            [
                                [['Inger', 'B-PER'], ['säger', '0'], .., []],
                            ]
        """
        sentences_rows_iob2 = list()
        for sentence in sentences_rows_iob1:
            sentence_iob2 = list()
            for i, row in enumerate(sentence):
                assert (
                    len(row) == 2
                ), f"ERROR! row = {row} should have length 0 or 2, not {len(row)}"
                current_tag = row[1]

                if (
                    current_tag == "O"
                    or "-" not in current_tag
                    or current_tag.startswith("B-")
                ):
                    sentence_iob2.append(row)
                elif current_tag.startswith("I-"):
                    previous_tag = (
                        sentence[i - 1][1]
                        if (i > 0 and len(sentence[i - 1]) == 2)
                        else None
                    )

                    if previous_tag not in [
                        current_tag,
                        current_tag.replace("I-", "B-"),
                    ]:
                        tag_iob2 = current_tag.replace("I-", "B-")
                        sentence_iob2.append([row[0], tag_iob2])
                    else:
                        sentence_iob2.append(row)

            sentences_rows_iob2.append(sentence_iob2)

        return sentences_rows_iob2

    @staticmethod
    def _shuffle_dataset(_phase: str, _sentences_rows: List[Any]) -> List[Any]:
        """
        III: format data

        Args:
            _phase: "train", "val", "test"
            _sentences_rows: e.g. (-pretokenized-)
                            [
                                [['Inger', 'PER'], ['säger', '0'], .., []],
                                [['Det', '0'], .., []]
                            ]
                            e.g. (-unpretokenized-)
                            [
                                {
                                    'text': 'Inger säger ..',
                                    'tags': [{'token': 'Inger', 'tag': 'PER', 'char_start': 0, 'char_end': 5}, ..],
                                },
                                {
                                    'text': 'Det ..',
                                    'tags': [{..}, ..]
                                }
                            ]

        Returns:
            _sentences_rows_shuffled: e.g. (-pretokenized-)
                            [
                                [['Det', '0'], .., [0],
                                [['Inger', 'PER'], ['säger', '0'], .., []]
                            ]
                            e.g. (-unpretokenized-)
                            [
                                {
                                    'text': 'Det ..',
                                    'tags': [{..}, ..]
                                },
                                {
                                    'text': 'Inger säger ..',
                                    'tags': [{'token': 'Inger', 'tag': 'PER', 'char_start': 0, 'char_end': 5}, ..],
                                }
                            ]
        """
        # change _sentences_rows by shuffling sentences
        random.Random(SEED_SHUFFLE[_phase]).shuffle(_sentences_rows)
        return _sentences_rows

    ####################################################################################################################
    # HELPER: READ FORMATTED
    ####################################################################################################################
    def _read_formatted_csvs(self, phases: List[str]) -> pd.DataFrame:
        """
        IV: resplit data

        Args:
            phases: .. to read formatted csvs from, e.g. ['val', 'test']

        Returns:
            df_phases: contains formatted csvs of phases
        """
        df_phases = [self._read_formatted_csv(phase) for phase in phases]
        return pd.concat(df_phases, ignore_index=True)

    def _read_formatted_csv(self, phase: str):
        """
        IV: resplit data

        Args:
            phase: .. to read formatted df from, e.g. 'val'

        Returns:
            df: formatted df
        """
        formatted_file_path = join(self.dataset_path, f"{phase}_formatted.csv")
        return pd.read_csv(formatted_file_path, sep="\t", header=None)

    ####################################################################################################################
    # HELPER: WRITE FINAL
    ####################################################################################################################
    @staticmethod
    def _split_off_validation_set(
        _df_original: pd.DataFrame, _val_fraction: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        IV: resplit data

        Args:
            _df_original:  df before splitting
            _val_fraction: between 0 and 1

        Returns:
            _df_new:       df after splitting, remainder
            _df_val:       df after splitting, validation
        """
        split_index = int(len(_df_original) * (1 - _val_fraction))
        _df_new = _df_original.iloc[:split_index]
        _df_val = _df_original.iloc[split_index:]
        return _df_new, _df_val

    def _write_final_csv(self, phase, df) -> None:  # pragma: no cover
        """
        IV: resplit data

        Args:
            phase: [str], 'train', 'val' or 'test'
            df: [pd DataFrame]

        Returns: -
        """
        file_path = join(self.dataset_path, f"{phase}.csv")
        df.to_csv(file_path, sep="\t", index=False, header=None)
