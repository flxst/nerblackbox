import os
import requests

from os.path import join, isfile
from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter


class CoNLL2003Formatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "conll2003"
        ner_tag_list = ["PER", "ORG", "LOC", "MISC"]
        super().__init__(ner_dataset, ner_tag_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def get_data(self, verbose: bool):
        """
        I: get data
        -----------
        :param verbose: [bool]
        :return: -
        """
        url_base = "https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/"
        targets = ["eng.train", "eng.testa", "eng.testb"]

        for target in targets:
            target_file = join(self.dataset_path, target)

            # fetch tgz from url
            if isfile(target_file):
                if verbose:
                    print(f".. file at {target_file} already exists")
            else:
                url = url_base + target
                myfile = requests.get(url, allow_redirects=True)
                open(target_file, "wb").write(myfile.content)
                if verbose:
                    print(f".. file fetched from {url} and saved at {target_file}")

    def create_ner_tag_mapping(self):
        """
        II: customize ner_training tag mapping if wanted
        -------------------------------------
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return dict()

    def format_data(self):
        """
        III: format data
        ----------------
        :return: -
        """
        for phase in ["train", "val", "test"]:
            rows = self._read_original_file(phase)
            self._write_formatted_csv(phase, rows)

    def resplit_data(self, val_fraction: float):
        """
        IV: resplit data
        ----------------
        :param val_fraction: [float]
        :return: -
        """
        # train -> train
        df_train = self._read_formatted_csvs(["train"])
        self._write_final_csv("train", df_train)

        # val  -> val
        df_val = self._read_formatted_csvs(["val"])
        self._write_final_csv("val", df_val)

        # test  -> test
        df_test = self._read_formatted_csvs(["test"])
        self._write_final_csv("test", df_test)

    ####################################################################################################################
    # HELPER: READ ORIGINAL
    ####################################################################################################################
    def _read_original_file(self, phase):
        """
        III: format data
        ---------------------------------------------
        :param phase:   [str] 'train' or 'test'
        :return: _rows: [list] of [list] of [str], e.g. [[], ['Inger', 'PER'], ['säger', '0'], ..]
        """
        file_name = {
            "train": "eng.train",
            "val": "eng.testa",
            "test": "eng.testb",
        }
        file_path_original = join(self.dataset_path, file_name[phase])

        _rows = list()
        if os.path.isfile(file_path_original):
            with open(file_path_original) as f:
                for i, row in enumerate(f.readlines()):
                    _rows.append(row.strip().split())
            print(f"\n> read {file_path_original}")

        _rows = [
            [row[0], row[-1]] if (len(row) == 4 and row[0] != "-DOCSTART-") else list()
            for row in _rows
        ]

        return _rows
