
import os
import pandas as pd
import json
from ner.data_preprocessing.tools.input_example import InputExample


class NerProcessor:
    """
    reads text data from csv and creates list of InputExamples
    """

    def __init__(self,
                 path,
                 tokenizer,
                 do_lower_case,
                 csv_file_separator='\t'):
        """
        :param path:               [str] to folder that contains dataset csv files (train, valid, test)
        :param tokenizer:          [transformers Tokenizer]
        :param do_lower_case:      [bool]
        :param csv_file_separator: [str], for datasets' csv files, e.g. '\t'
        """
        # input arguments
        self.path = path
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case
        self.csv_file_separator = csv_file_separator

        # additional attributes
        self.token_count = None

        # processing
        with open(os.path.join(self.path, 'ner_tag_mapping.json'), 'r') as f:
            self.ner_tag_mapping = json.load(f)

        self.data = dict()
        for phase in ['train', 'valid', 'test']:
            self.data[phase] = self._read_csv(os.path.join(self.path, f'{phase}.csv'))

    ####################################################################################################################
    # PUBLIC METHODS
    ####################################################################################################################
    def get_input_examples(self, phase):
        """
        gets list of input examples for specified phase
        -----------------------------------------------
        :param phase: [str], e.g. 'train', 'valid', 'test'
        :return: [list] of [InputExample]
        """
        return self._create_list_of_input_examples(self.data[phase], phase)

    def get_tag_list(self):
        """
        get tag list derived from ner_tag_mapping
        ---------------------------------------------
        :return: [list] of [str]
        """
        return ['[PAD]', '[CLS]', '[SEP]'] + list(set(self.ner_tag_mapping.values()))

    ####################################################################################################################
    # PRIVATE METHODS
    ####################################################################################################################
    def _read_csv(self, path):
        """
        read csv using pandas.

        Note: The csv is expected to
        - have two columns seperated by self.seperator
        - not have a header with column names
        ----------------------------------------------
        :param path: [str]
        :return: [pandas dataframe]
        """
        return pd.read_csv(path, names=['tags', 'text'], header=None, sep=self.csv_file_separator)

    def _create_list_of_input_examples(self, df, set_type):
        """
        create list of input examples from pandas dataframe created from _read_csv() method
        -----------------------------------------------------------------------------------
        :param df:                 [pandas dataframe] with columns 'tags', 'text'
        :param set_type:           [str], e.g. 'train', 'valid', 'test'
        :changed attr: token_count [int] total number of tokens in df
        :return: [list] of [InputExample]
        """
        self.token_count = 0

        examples = []
        for i, row in enumerate(df.itertuples()):
            # input_example
            guid = f'{set_type}-{i}'
            text_a = row.text.lower() if self.do_lower_case else row.text
            tags_a = row.tags

            input_example = InputExample(guid=guid,
                                         text_a=text_a,
                                         tags_a=tags_a)

            # append
            examples.append(input_example)
        return examples