import os

from os.path import join
from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter


class SUCFormatter(BaseFormatter):
    def __init__(self):
        ner_dataset = "suc"
        ner_tag_list = [
            "person",
            "place",
            "inst",
            "work",
            "misc",
        ]  # misc: see create_ner_tag_mapping()
        super().__init__(ner_dataset, ner_tag_list)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def get_data(self, verbose: bool):
        """
        I: get data
        ----------
        :param verbose: [bool]
        :return: -
        """
        print("(suc: nothing to do)")

    def create_ner_tag_mapping(self):
        """
        II: customize ner_training tag mapping if wanted
        ------------------------------------------------
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        return dict()
        # return {
        #     'animal': 'misc',
        #     'myth': 'misc',
        #     'product': 'misc',
        #     'event': 'misc',
        #     'other': 'misc',
        # }

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
            "train": "suc-train.conll",
            "val": "suc-dev.conll",
            "test": "suc-test.conll",
        }
        file_path_original = join(self.dataset_path, file_name[phase])

        _rows = list()
        if os.path.isfile(file_path_original):
            with open(file_path_original) as f:
                for i, row in enumerate(f.readlines()):
                    _rows.append(row.strip().split())
            print(f"\n> read {file_path_original}")
        else:
            raise Exception(f"> original file {file_path_original} could not be found.")

        _rows = [
            [row[1], self.transform_tags(row[-3], row[-2])] if len(row) > 0 else list()
            for row in _rows
        ]

        return _rows

    @staticmethod
    def transform_tags(bio, tag):
        """
        :param bio: [str] 'O', 'B', 'I'
        :param tag: [str] '_', 'person', ..
        :return: transformed tag: [str], e.g. 'O', 'B-person', ..
        """
        if bio == "O":
            return "O"
        else:
            return f"{bio}-{tag}"
