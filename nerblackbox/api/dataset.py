import os
import csv
from os.path import isfile
import json
from typing import Optional
from nerblackbox import Store
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter

PHASES = ["train", "val", "test"]


class Dataset:
    r"""
    class to download, set up and inspect a single dataset
    """
    path = os.environ.get("DATA_DIR")

    def __init__(
        self,
        name: str,
        source: str,
        pretokenized: bool = False,
        split: bool = False,
        file_path: Optional[str] = None,
        subset: Optional[str] = None,
    ):
        r"""
        Args:
             name: name of dataset, e.g. "swedish_ner_corpus"
             source: source of dataset, e.g. "HF", "BI", "LF"
             pretokenized: [only for source = "LF"] whether the dataset is pretokenized. otherwise, it has the standard type.
             split: [only for source = "LF"] whether the dataset is split into train/val/test subsets. otherwise, it is a single file.
             file_path: [only for source = "LF"] absolute file_path
             subset: [only for source = "HF"] name of subset if applicable, e.g. "simple_cased"
        """
        assert source in ["HF", "BI", "LF"], f"ERROR! source = {source} needs to be HF, BI or LF."

        self.dataset_name = name
        self.source = source
        self.dataset_subset_name = subset if subset is not None else ""
        self.pretokenized = pretokenized
        self.split = split

        self.file_extension = "csv" if self.pretokenized else "jsonl"

        self.file_path: Optional[str] = None
        if self.source == "LF":
            if file_path is None:
                self.file_path = \
                    f"{Store.get_path()}/datasets/{self.dataset_name}/{self.dataset_name}.{self.file_extension}"
            else:
                self.file_path = file_path
            self._assert_file_existence()

    def _assert_file_existence(self) -> None:
        r"""
        check that self.file_path exists, throw error if not
        """
        assert isinstance(self.file_path, str), f"ERROR! type(self.file_path) = {type(self.file_path)}. Expected str."
        if not isfile(self.file_path):
            raise Exception(f"ERROR! {self.file_path} does not exist")

    def set_up(self, val_fraction: Optional[float] = None, test_fraction: Optional[float] = None) -> None:
        r"""
        sets up the dataset and creates the following files (if needed):

        - `<STORE_DIR>/datasets/<name>/train.*`
        - `<STORE_DIR>/datasets/<name>/val.*`
        - `<STORE_DIR>/datasets/<name>/test.*`

        where `* = jsonl` or `* = csv`, depending on whether the data is pretokenized or not.

        Args:
            val_fraction: e.g. 0.1 (applicable if source = HF, BI, LF)
            test_fraction: e.g. 0.1 (applicable if source = LF)
        """
        if self.source in ["HF", "BI"]:
            if val_fraction is None:
                val_fraction = 0.111111111
            self._set_up_remote(val_fraction=val_fraction)
        else:  # LF
            if not self.split:
                if val_fraction is None:
                    val_fraction = 0.1
                if test_fraction is None:
                    test_fraction = 0.1
                self._split(val_fraction=val_fraction, test_fraction=test_fraction)

    def _split(self, val_fraction: float, test_fraction: float) -> None:
        r"""
        method only for source = LF

        - read dataset from self.file_path
        - split dataset into train/val/test subsets
        - write subsets to PHASE.jsonl or PHASE.csv

        Args:
            val_fraction: e.g. 0.1
            test_fraction: e.g. 0.1
        """
        assert isinstance(
            self.file_path, str
        ), f"ERROR! type(self.file_path) = {self.file_path} should be str."

        with open(self.file_path, "r") as f:
            if self.pretokenized:
                reader = csv.reader(f)
                input_lines = [row for row in reader]
            else:
                input_lines = [json.loads(line) for line in f]
        print(f"> read {len(input_lines)} documents from {self.file_path}")

        index_train_end = int((1 - val_fraction - test_fraction) * len(input_lines))
        index_val_end = int((1 - test_fraction) * len(input_lines))
        output_lines = {
            "train": input_lines[:index_train_end],
            "val": input_lines[index_train_end:index_val_end],
            "test": input_lines[index_val_end:],
        }

        print()
        for phase in PHASES:
            file_path_phase = (
                "/".join(self.file_path.split("/")[:-1]) + f"/{phase}.{self.file_extension}"
            )

            with open(file_path_phase, "w") as f:
                for line in output_lines[phase]:
                    if self.pretokenized:
                        writer = csv.writer(f)
                        writer.writerow(line)
                    else:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

            print(f"> wrote {len(output_lines[phase])} documents to {file_path_phase}")

    def _set_up_remote(
        self,
        val_fraction: float,
        modify: bool = True,
        verbose: bool = False,
        shuffle: bool = False,
    ) -> None:
        r"""
        method only for source = HF, BI

        downloads and sets up the dataset. creates the following files:

            <STORE_DIR>/datasets/<dataset_name>/train.*
            <STORE_DIR>/datasets/<dataset_name>/val.*
            <STORE_DIR>/datasets/<dataset_name>/test.*

        where `* = json` or `* = csv`, depending on whether the data is pretokenized or not

        Args:
            modify: if True: modify tags as specified in method modify_ner_tag_mapping()  TODO: better explanation
            val_fraction: fraction of the validation dataset if it is split off from the training dataset
            verbose: verbose output
            shuffle: whether to shuffle train/val/test datasets
        """
        _parameters = {
            "ner_dataset": self.dataset_name,
            "ner_dataset_subset": self.dataset_subset_name,
            "modify": modify,
            "val_fraction": val_fraction,
            "verbose": verbose,
        }

        formatter = AutoFormatter.for_dataset(
            self.dataset_name, self.dataset_subset_name, verbose
        )
        formatter.create_directory()
        formatter.get_data(verbose=verbose)  # I: get_data
        formatter.create_ner_tag_mapping_json(
            modify=modify,
            verbose=verbose
        )  # II: create ner tag mapping
        formatter.format_data(shuffle=shuffle)  # III: format data
        formatter.resplit_data(
            val_fraction=val_fraction, write_csv=True
        )  # IV: resplit data
        formatter.analyzer.analyze_data()  # V: analyze data
        formatter.analyzer.plot_data()  # V: analyze data

    @staticmethod
    def overview():
        print("Dataset.overview() not yet implemented.")  # TODO

    def _analyze(self):
        r"""
        analyze the dataset
        """
        formatter = AutoFormatter.for_dataset(self.dataset_name)
        formatter.analyzer.analyze_data()  # V: analyze data
        formatter.analyzer.plot_data()  # V: analyze data
