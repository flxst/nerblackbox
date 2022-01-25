from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from os.path import join
from datasets import (
    load_dataset,
    load_dataset_builder,
    get_dataset_split_names,
    DatasetDict,
)
from shutil import copyfile

from nerblackbox.modules.datasets.formatter.base_formatter import (
    BaseFormatter,
    SENTENCES_ROWS,
    SENTENCES_ROWS_PRETOKENIZED,
    SENTENCES_ROWS_UNPRETOKENIZED,
)


class HuggingfaceDatasetsFormatter(BaseFormatter):

    PHASES = ["train", "val", "test"]
    PHASES_DATASETS = {"train": "train", "val": "validation", "test": "test"}

    @classmethod
    def check_existence(cls, ner_dataset: str) -> bool:
        """
        checks if ner_dataset exists in huggingface datasets

        Args:
            ner_dataset: e.g. "conll2003"

        Returns:
            existence: True if ner_dataset exists in huggingface datasets, False otherwise
        """
        try:
            _ = load_dataset_builder(ner_dataset)
            return True
        except FileNotFoundError:
            return False

    @classmethod
    def check_compatibility(cls, ner_dataset: str) -> bool:
        """
        checks if ner_dataset contains train/val/test splits

        Args:
            ner_dataset: e.g. "conll2003"

        Returns:
            compatibility: True if ner_dataset contains train/val/test splits, False otherwise
        """
        _dataset_split_names = get_dataset_split_names(ner_dataset)
        return sorted(_dataset_split_names) == ["test", "train", "validation"]

    @classmethod
    def check_implementation(cls, ner_dataset: str) -> bool:
        """
        problem: there is no common structure in dataset_builder.info.features for all datasets
        parsing for a few typical structures is implemented
        this method checks if one of them is applicable to ner_dataset

        Args:
            ner_dataset: e.g. "conll2003"

        Returns:
            implementation: True if ner_dataset is implemented, False otherwise
        """
        implementation, _, _, _ = cls.get_infos(ner_dataset)
        return implementation

    @classmethod
    def get_infos(
        cls, ner_dataset: str
    ) -> Tuple[bool, Optional[List[str]], Optional[bool], Optional[Dict[str, Any]]]:
        """
        get all relevant infos about dataset

        Args:
            ner_dataset: e.g. "conll2003"

        Returns:
            implementation: True if ner_dataset is implemented, False otherwise
            tags: e.g. ["O", "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER"]
            pretokenized: e.g. True
            lookup_table: e.g. {'text': 'tokens', 'tags': 'ner_tags', 'mapping': None}
                          e.g. {'text': 'sentence', 'tags': 'entities', 'mapping': {..}}
        """
        dataset_builder = load_dataset_builder(ner_dataset)
        if dataset_builder.info.features is None:
            return False, None, None, None
        else:
            feat = dict(dataset_builder.info.features)

        implementation: bool = False
        tags: Optional[List[str]] = None
        pretokenized: Optional[bool] = None
        lookup_table: Optional[Dict[str, Any]] = None
        try:
            if "ner_tags" in feat:  # e.g. conll2003
                keys = ["tokens", "ner_tags"]
                if all([key in feat for key in keys]):
                    implementation = True
                    tags = feat["ner_tags"].feature.names
                    pretokenized = True
                    lookup_table = {
                        "text": "tokens",
                        "tags": "ner_tags",
                        "mapping": None,
                    }
            elif "entities" in feat:  # e.g. ehealth_kd
                keys = ["sentence", "entities"]
                if all([key in feat for key in keys]):
                    entities_keys = [
                        "ent_text",
                        "ent_label",
                        "start_character",
                        "end_character",
                    ]
                    if all(
                        [
                            entities_key in feat["entities"][0]
                            for entities_key in entities_keys
                        ]
                    ):
                        implementation = True
                        tags = feat["entities"][0]["ent_label"].names
                        pretokenized = False
                        lookup_table = {
                            "text": "sentence",
                            "tags": "entities",
                            "mapping": {
                                "ent_text": "token",
                                "ent_label": "tag",
                                "start_character": "char_start",
                                "end_character": "char_end",
                            },
                        }
            else:
                return False, None, None, None
        except Exception:
            return False, None, None, None

        if (
            implementation is False
            or tags is None
            or pretokenized is None
            or lookup_table is None
        ):
            return False, None, None, None
        else:
            return implementation, tags, pretokenized, lookup_table

    def __init__(self, ner_dataset: str):
        _, self.tags, self.pretokenized, self.lookup_table = self.get_infos(ner_dataset)
        self.sentences_rows_pretokenized: Dict[str, SENTENCES_ROWS_PRETOKENIZED] = {
            phase: list() for phase in self.PHASES
        }
        self.sentences_rows_unpretokenized: Dict[str, SENTENCES_ROWS_UNPRETOKENIZED] = {
            phase: list() for phase in self.PHASES
        }
        super().__init__(ner_dataset, ner_tag_list=self.get_ner_tag_list())

    def get_ner_tag_list(self) -> List[str]:
        """
        reduces tags to plain tags

        Used attr:
            tags: e.g. ["O", "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER"]

        Returns:
            ner_tag_list: ordered, e.g.["LOC", "MISC", "ORG", "PER"]
        """
        assert self.tags is not None, f"ERROR! self.tags unexpectedly found to be None."
        return sorted(
            list(set([tag.split("-")[-1] for tag in self.tags if tag != "O"]))
        )

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def get_data(self, verbose: bool) -> None:  # pragma: no cover
        """
        I: get data

        Args:
            verbose: [bool]

        Created attr:
            sentences_rows: e.g. (-pretokenized-)
                            [
                                [['Inger', 'PER'], ['säger', '0'], .., []],
                                [['Det', '0'], .., []]
                            ]
                            e.g. (-not pretokenized-)
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
        """
        # typing
        assert self.tags is not None, f"ERROR! self.tags unexpectedly found to be None."
        assert (
            self.lookup_table is not None
        ), f"ERROR! self.lookup_table unexpectedly found to be None."

        # start
        text = self.lookup_table["text"]
        tags = self.lookup_table["tags"]
        dataset = load_dataset(self.ner_dataset)
        assert isinstance(
            dataset, DatasetDict
        ), f"ERROR! type(dataset) = {type(dataset)} should be DatasetDict"
        for phase in self.PHASES:
            phase_dataset = self.PHASES_DATASETS[phase]
            for field in [text, tags]:
                assert (
                    field in dataset[phase_dataset].info.features.keys()
                ), f"ERROR! field = {field} not present in dataset."
            for text_sentence, tags_sentence in zip(
                dataset[phase_dataset][:][text], dataset[phase_dataset][:][tags]
            ):
                if self.pretokenized:
                    self.sentences_rows_pretokenized[phase].append(
                        [
                            [text_single, self.tags[int(tag_single)]]
                            for text_single, tag_single in zip(
                                text_sentence, tags_sentence
                            )
                        ]
                    )
                else:
                    _dict = {
                        "text": text_sentence,
                        "tags": [
                            {
                                self.lookup_table["mapping"][key]: value
                                for key, value in tags_sentence[n].items()
                                if key in self.lookup_table["mapping"]
                            }
                            for n in range(len(tags_sentence))
                        ],
                    }
                    for tag_dict in _dict["tags"]:
                        tag_dict.update(
                            (k, self.tags[int(v)])
                            for k, v in tag_dict.items()
                            if k == "tag"
                        )

                    # there are multiword entities that are not connected
                    # (i.e. there are other words between the entities' tokens)
                    # e.g. ehealth_kd: {"token": "uno días", "tag": "Concept", "char_start": 64170, "char_end": 64183}
                    # -> filter them out
                    _dict["tags"] = [
                        elem
                        for elem in _dict["tags"]
                        if len(elem["token"]) == elem["char_end"] - elem["char_start"]
                    ]

                    self.sentences_rows_unpretokenized[phase].append(_dict)

    def create_ner_tag_mapping(self) -> Dict[str, str]:
        """
        II: customize ner_training tag mapping if wanted

        Returns:
            ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return dict()

    def format_data(
        self, shuffle: bool = True, write_csv: bool = True
    ) -> Optional[SENTENCES_ROWS]:
        """
        III: format data

        Args:
            shuffle: whether to shuffle rows of dataset
            write_csv: whether to write dataset to csv (should always be True except for testing)

        Returns:
            sentences_rows_iob2: only if write_csv = False
        """
        if self.pretokenized:
            for phase in ["train", "val", "test"]:
                sentences_rows_pretokenized_phase = self.sentences_rows_pretokenized[
                    phase
                ].copy()
                if shuffle:
                    sentences_rows_pretokenized_phase = self._shuffle_dataset(
                        phase, sentences_rows_pretokenized_phase
                    )

                sentences_rows_iob2 = self._convert_iob1_to_iob2(
                    sentences_rows_pretokenized_phase
                )
                if write_csv:  # pragma: no cover
                    self._write_formatted_csv(phase, sentences_rows_iob2)
                else:
                    return sentences_rows_iob2  # return train!
        else:
            for phase in ["train", "val", "test"]:
                sentences_rows_unpretokenized_phase = (
                    self.sentences_rows_unpretokenized[phase].copy()
                )
                if shuffle:
                    sentences_rows_unpretokenized_phase = self._shuffle_dataset(
                        phase, sentences_rows_unpretokenized_phase
                    )

                if write_csv:  # pragma: no cover
                    self._write_formatted_jsonl(
                        phase, sentences_rows_unpretokenized_phase
                    )
                else:
                    return sentences_rows_unpretokenized_phase  # returns train!
        return None

    def set_original_file_paths(self) -> None:  # pragma: no cover
        """
        III: format data

        Changed Attributes:
            file_paths: [Dict[str, str]], e.g. {'train': <path_to_train_csv>, 'val': ..}

        Returns: -
        """
        pass  # not necessary as _read_original_file() is not used

    def _parse_row(self, _row: str) -> List[str]:  # pragma: no cover
        """
        III: format data

        Args:
            _row: e.g. "Det PER X B"

        Returns:
            _row_list: e.g. ["Det", "PER", "X", "B"]
        """
        pass  # not necessary as _read_original_file() is not used

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
        pass  # not necessary as _read_original_file() is not used

    def resplit_data(
        self, val_fraction: float = 0.0, write_csv: bool = True
    ) -> Optional[Tuple[pd.DataFrame, ...]]:
        """
        IV: resplit data

        Args:
            val_fraction: [float], e.g. 0.3
            write_csv: whether to write dataset to csv (should always be True except for testing)

        Returns:
            df_train: only if write_csv = False
            df_val:   only if write_csv = False
            df_test:  only if write_csv = False
        """
        if self.pretokenized:
            # train -> train
            df_train = self._read_formatted_csvs(["train"])

            # val  -> val
            df_val = self._read_formatted_csvs(["val"])

            # test  -> test
            df_test = self._read_formatted_csvs(["test"])

            if write_csv:  # pragma: no cover
                self._write_final_csv("train", df_train)
                self._write_final_csv("val", df_val)
                self._write_final_csv("test", df_test)
                return None
            else:
                return df_train, df_val, df_test
        else:  # pragma: no cover
            for phase in self.PHASES:
                src = join(self.dataset_path, f"{phase}_formatted.jsonl")
                dst = join(self.dataset_path, f"{phase}.jsonl")
                copyfile(src, dst)
            return None
