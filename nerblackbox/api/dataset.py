import os
from os.path import isfile
import json
from typing import Optional
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter

PHASES = ["train", "val", "test"]


class Dataset:
    r"""
    class to download, set up and inspect a single dataset

    Attributes: see `__init__()`
    """
    path = os.environ.get("DATA_DIR")

    def __init__(
        self,
        dataset_name: str,
        dataset_subset_name: str = "",
    ):
        r"""
        Args:
             dataset_name: name of dataset, e.g. "swedish_ner_corpus"
             dataset_subset_name: name of subset if applicable (for HuggingFace Datasets), e.g. "simple_cased"
        """
        self.dataset_name = dataset_name
        self.dataset_subset_name = dataset_subset_name
        self.file_path = None  # only for instances created through from_file method

    @classmethod
    def from_file(cls,
                  dataset_name: str,
                  file_path: str) -> Optional['Dataset']:
        r"""
        create a Dataset instance by reading a file

        Args:
            dataset_name: name of dataset, e.g. "strangnas"
            file_path: e.g. "strangnas/strangnas.jsonl"

        Returns:
            dataset instance

        """
        if isfile(file_path):
            print(f"> create Dataset")
            dataset = Dataset(dataset_name)
            dataset.file_path = file_path
            return dataset
        else:
            print(f"Error! {file_path} does not exist")
            return None

    def split(self,
              val_fraction: float = 0.1,
              test_fraction: float = 0.2) -> None:
        r"""
        - read dataset from self.file_path
        - split dataset into train/val/test subsets
        - write subsets to PHASE.jsonl

        Args:
            val_fraction: e.g. 0.1
            test_fraction: e.g. 0.2
        """
        with open(self.file_path, "r") as f:
            input_lines = [json.loads(line) for line in f]
        print(f"> read {len(input_lines)} documents from {self.file_path}")

        index_train_end = (1-val_fraction-test_fraction)*len(input_lines)
        index_val_end = (1-test_fraction)*len(input_lines)
        output_lines = {
            "train": input_lines[:index_train_end],
            "val": input_lines[index_train_end: index_val_end],
            "test": input_lines[index_val_end:],
        }

        print()
        for phase in PHASES:
            file_path_phase = "/".join(self.file_path.split("/")[:-1]) + f"/{phase}.jsonl"

            with open(file_path_phase, "w") as f:
                for line in output_lines[phase]:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            print(f"> wrote {len(output_lines[phase])} documents to {file_path_phase}")

    def set_up(
        self,
        modify: bool = True,
        val_fraction: float = 0.3,
        verbose: bool = False,
    ):
        r"""downloads and sets up the dataset. creates the following files:

            <STORE_DIR>/datasets/<dataset_name>/train.*
            <STORE_DIR>/datasets/<dataset_name>/val.*
            <STORE_DIR>/datasets/<dataset_name>/test.*

        where `* = json` or `* = csv`, depending on whether the data is pretokenized or not

        Args:
            modify: if True: modify tags as specified in method modify_ner_tag_mapping()  TODO: better explanation
            val_fraction: fraction of the validation dataset if it is split off from the training dataset
            verbose: verbose output
        """
        _parameters = {
            "ner_dataset": self.dataset_name,
            "ner_dataset_subset": self.dataset_subset_name,
            "modify": modify,
            "val_fraction": val_fraction,
            "verbose": verbose,
        }

        formatter = AutoFormatter.for_dataset(self.dataset_name, self.dataset_subset_name)
        formatter.create_directory()
        formatter.get_data(verbose=verbose)  # I: get_data
        formatter.create_ner_tag_mapping_json(
            modify=modify
        )  # II: create ner tag mapping
        formatter.format_data(shuffle=True)  # III: format data
        formatter.resplit_data(val_fraction=val_fraction, write_csv=True)  # IV: resplit data
        formatter.analyzer.analyze_data()  # V: analyze data
        formatter.analyzer.plot_data()  # V: analyze data

    @staticmethod
    def overview():
        print("Dataset.overview() not yet implemented.")

    def _analyze(self):
        r"""
        analyze the dataset
        """
        formatter = AutoFormatter.for_dataset(self.dataset_name)
        formatter.analyzer.analyze_data()  # V: analyze data
        formatter.analyzer.plot_data()  # V: analyze data
