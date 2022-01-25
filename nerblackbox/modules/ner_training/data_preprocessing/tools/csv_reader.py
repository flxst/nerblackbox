import os
from typing import List, Dict
import pandas as pd
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)


class CsvReader:
    """
    reads data (tags & text) from csv and
    - creates list of InputExamples
    - gets list of tags
    """

    def __init__(
        self,
        path,
        tokenizer,
        pretokenized,
        do_lower_case,
        default_logger=None,
        csv_file_separator="\t",
    ):
        """
        :param path:               [str] to folder that contains dataset csv files (train, val, test)
        :param tokenizer:          [transformers Tokenizer]
        :param pretokenized:       [bool]
        :param do_lower_case:      [bool]
        :param default_logger:     []
        :param csv_file_separator: [str], for datasets' csv files, e.g. '\t'
        """
        # input arguments
        self.path = path
        self.tokenizer = tokenizer
        self.pretokenized = pretokenized
        self.do_lower_case = do_lower_case
        self.default_logger = default_logger
        self.csv_file_separator = csv_file_separator

        # additional attributes
        self.token_count = None

        # data & annotation_classes
        self.data: Dict[str, pd.DataFrame] = dict()
        self.annotation_classes: List[str] = list()

        # process
        self._process()

    ####################################################################################################################
    # PUBLIC METHODS
    ####################################################################################################################
    def get_input_examples(self, phase):
        """
        gets list of input examples for specified phase
        -----------------------------------------------
        :param phase: [str], e.g. 'train', 'val', 'test'
        :return: [list] of [InputExample]
        """
        return self._create_list_of_input_examples(self.data[phase], phase)

    ####################################################################################################################
    # PRIVATE METHODS
    ####################################################################################################################
    def _process(self):
        """
        read csv, get data and annotation_classes

        Created attr:
            data: [dict] w/ key = phase & value = [pandas df]
            annotation_classes: [list] of [str], e.g. ["O", "B-PER", "I-PER", ..]

        Returns: -
        """
        annotation_classes_found = list()
        for phase in ["train", "val", "test"]:
            # data
            self.data[phase] = self._read_csv(
                os.path.join(
                    self.path,
                    f"{phase}.csv"
                    if self.pretokenized
                    else f"pretokenized_{phase}.csv",
                )
            )

            # tag list
            annotation_classes_phase = list(
                set(" ".join(self.data[phase]["tags"].values).split())
            )
            annotation_classes_found = sorted(
                list(set(annotation_classes_found + annotation_classes_phase))
            )

        self.annotation_classes: List[str] = ["O"] + [
            elem for elem in annotation_classes_found if elem != "O"
        ]

        if self.default_logger:
            self.default_logger.log_debug(
                f"> tag list found in data: {annotation_classes_found}"
            )
            self.default_logger.log_debug(
                f"> tag list complete:      {self.annotation_classes}"
            )

    def _read_csv(self, path: str) -> pd.DataFrame:
        """
        read csv using pandas.

        Note: The csv is expected to
        - have two columns seperated by self.seperator
        - not have a header with column names
        ----------------------------------------------
        :param path: [str]
        :return: [pandas dataframe]
        """
        return pd.read_csv(
            path, names=["tags", "text"], header=None, sep=self.csv_file_separator
        )

    def _create_list_of_input_examples(self, df, set_type):
        """
        create list of input examples from pandas dataframe created from _read_csv() method
        -----------------------------------------------------------------------------------
        :param df:                 [pandas dataframe] with columns 'tags', 'text'
        :param set_type:           [str], e.g. 'train', 'val', 'test'
        :changed attr: token_count [int] total number of tokens in df
        :return: [list] of [InputExample]
        """
        self.token_count = 0

        examples = []
        for i, row in enumerate(df.itertuples()):
            # input_example
            guid = f"{set_type}-{i}"
            text = row.text.lower() if self.do_lower_case else row.text
            tags = row.tags

            input_example = InputExample(guid=guid, text=text, tags=tags)

            # append
            examples.append(input_example)
        return examples
