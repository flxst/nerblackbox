
import pandas as pd
from abc import ABC, abstractmethod


class BaseFormatter(ABC):

    def __init__(self, ner_dataset, ner_label_list):
        """
        :param ner_dataset:    [str] 'swedish_ner_corpus' or 'SUC'
        :param ner_label_list: [list] of [str], e.g. ['PER', 'LOC', ..]
        """
        self.ner_dataset = ner_dataset
        self.ner_label_list = ner_label_list

    def create_ner_label_mapping(self, with_tags: bool, modify: bool):
        """
        create customized ner label mapping to map labels in original data to labels in formatted data
        ----------------------------------------------------------------------------------------------
        :param with_tags: [bool], if True: create labels with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple labels, e.g. 'PER', 'LOC', ..
        :param modify:    [bool], if True: modify labels as specified in method modify_ner_label_mapping()
        :return: ner_label_mapping: [dict] w/ keys = labels in original data, values = labels in formatted data
        """
        # full label list
        if with_tags:
            _label_lists_extended = [[f'B-{label}', f'I-{label}'] for label in self.ner_label_list]
            label_list_full = ['O'] + [l_i for l in _label_lists_extended for l_i in l]
        else:
            label_list_full = ['O'] + self.ner_label_list

        # map each label to itself
        ner_label_mapping_original = {k: k for k in label_list_full}

        if modify:
            return self.modify_ner_label_mapping(ner_label_mapping_original, with_tags=with_tags)
        else:
            return ner_label_mapping_original

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def modify_ner_label_mapping(self, ner_label_mapping_original, with_tags: bool):
        """
        customize ner label mapping if wanted
        -------------------------------------
        :param ner_label_mapping_original: [dict] w/ keys = labels in original data, values = labels in original data
        :param with_tags: [bool], if True: create labels with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple labels, e.g. 'PER', 'LOC', ..
        :return: ner_label_mapping: [dict] w/ keys = labels in original data, values = labels in formatted data
        """
        pass

    @abstractmethod
    def read_original_file(self, phase):
        pass

    @abstractmethod
    def write_formatted_csv(self, phase, rows, dataset_path):
        pass

    ####################################################################################################################
    # BASE METHODS
    ####################################################################################################################
    def read_formatted_csv(self, phase):
        """
        - read formatted csv files
        ----------------------------------------------
        :param phase:         [str] 'train' or 'test'
        :return: num_sentences:    [int]
                 stats_aggregated: [pandas Series] with indices = labels, values = number of occurrences
        """
        file_path = f'datasets/ner/{self.ner_dataset}/{phase}.csv'

        df = pd.read_csv(file_path, sep='\t')
        # print(len(df), len(df.columns))
        # print(df.head())

        columns = ['O'] + self.ner_label_list
        stats = pd.DataFrame([], columns=columns)
        labels = df.iloc[:, 0].apply(lambda x: x.split())
        # print(labels.head())

        for column in columns:
            stats[column] = labels.apply(lambda x: len([elem for elem in x if elem == column]))
        # print(stats.head())

        assert len(df) == len(stats)

        num_sentences = len(stats)
        stats_aggregated = stats.sum().to_frame()
        stats_aggregated.columns = ['count']

        return num_sentences, stats_aggregated

    @staticmethod
    def stats_aggregated_add_columns(df, number_of_sentences):
        df['count/sentence'] = df['count']/float(number_of_sentences)
        df['count/sentence'] = df['count/sentence'].apply(lambda x: '{:.2f}'.format(x))

        number_of_filtered_occurrences = df['count'].sum() - df.loc['O']['count']
        df['relative'] = df['count'] / number_of_filtered_occurrences
        df['relative']['O'] = 0
        df['relative'] = df['relative'].apply(lambda x: '{:.2f}'.format(x))

        return df

