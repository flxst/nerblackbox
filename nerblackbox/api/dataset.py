import os
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter


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
