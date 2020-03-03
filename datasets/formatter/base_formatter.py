
import pandas as pd
from abc import ABC, abstractmethod


class BaseFormatter(ABC):

    def __init__(self, ner_dataset, ner_tag_list):
        """
        :param ner_dataset:  [str] 'swedish_ner_corpus' or 'SUC'
        :param ner_tag_list: [list] of [str], e.g. ['PER', 'LOC', ..]
        """
        self.ner_dataset = ner_dataset
        self.ner_tag_list = ner_tag_list

    def create_ner_tag_mapping(self, with_tags: bool, modify: bool):
        """
        create customized ner tag mapping to map tags in original data to tags in formatted data
        ----------------------------------------------------------------------------------------------
        :param with_tags: [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple tags, e.g. 'PER', 'LOC', ..
        :param modify:    [bool], if True: modify tags as specified in method modify_ner_tag_mapping()
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        # full tag list
        if with_tags:
            _tag_lists_extended = [[f'B-{tag}', f'I-{tag}'] for tag in self.ner_tag_list]
            tag_list_full = ['O'] + [l_i for l in _tag_lists_extended for l_i in l]
        else:
            tag_list_full = ['O'] + self.ner_tag_list

        # map each tag to itself
        ner_tag_mapping_original = {k: k for k in tag_list_full}

        if modify:
            return self.modify_ner_tag_mapping(ner_tag_mapping_original, with_tags=with_tags)
        else:
            return ner_tag_mapping_original

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def modify_ner_tag_mapping(self, ner_tag_mapping_original, with_tags: bool):
        """
        customize ner tag mapping if wanted
        -------------------------------------
        :param ner_tag_mapping_original: [dict] w/ keys = tags in original data, values = tags in original data
        :param with_tags: [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple tags, e.g. 'PER', 'LOC', ..
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
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
                 stats_aggregated: [pandas Series] with indices = tags, values = number of occurrences
        """
        file_path = f'datasets/ner/{self.ner_dataset}/{phase}.csv'

        df = pd.read_csv(file_path, sep='\t')

        columns = ['O'] + self.ner_tag_list
        stats = pd.DataFrame([], columns=columns)
        tags = df.iloc[:, 0].apply(lambda x: x.split())

        for column in columns:
            stats[column] = tags.apply(lambda x: len([elem for elem in x if elem == column]))

        assert len(df) == len(stats)

        num_sentences = len(stats)
        stats_aggregated = stats.sum().to_frame()
        stats_aggregated.columns = ['count']

        return num_sentences, stats_aggregated

    @staticmethod
    def stats_aggregated_add_columns(df, number_of_sentences):
        df['count/sentence'] = df['count']/float(number_of_sentences)
        df['count/sentence'] = df['count/sentence'].apply(lambda x: '{:.2f}'.format(x))

        # relative count w/ 0
        number_of_occurrences = df['count'].sum()
        df['relative count w/ 0'] = df['count'] / number_of_occurrences
        df['relative count w/ 0'] = df['relative count w/ 0'].apply(lambda x: '{:.2f}'.format(x))

        # relative count w/o 0
        number_of_filtered_occurrences = df['count'].sum() - df.loc['O']['count']
        df['relative count w/o 0'] = df['count'] / number_of_filtered_occurrences
        df['relative count w/o 0']['O'] = 0
        df['relative count w/o 0'] = df['relative count w/o 0'].apply(lambda x: '{:.2f}'.format(x))

        return df
