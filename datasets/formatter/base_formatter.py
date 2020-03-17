
import os
import json
import pandas as pd
from abc import ABC, abstractmethod

from os.path import abspath, dirname, join
import sys
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
sys.path.append(BASE_DIR)

from utils.utils import get_dataset_path
from datasets.plots import Plots

from utils.logger import get_file_logger


class BaseFormatter(ABC):

    def __init__(self, ner_dataset, ner_tag_list):
        """
        :param ner_dataset:  [str] 'swedish_ner_corpus' or 'SUC'
        :param ner_tag_list: [list] of [str], e.g. ['PER', 'LOC', ..]
        """
        self.ner_dataset = ner_dataset
        self.ner_tag_list = ner_tag_list
        self.dataset_path = join(BASE_DIR, get_dataset_path(ner_dataset))
        self.stats_aggregated = None

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def get_data(self, verbose: bool):
        """
        I: get data
        -----------
        :param verbose: [bool]
        :return: -
        """
        pass

    @abstractmethod
    def modify_ner_tag_mapping(self, ner_tag_mapping_original, with_tags: bool):
        """
        II: customize ner tag mapping if wanted
        -------------------------------------
        :param ner_tag_mapping_original: [dict] w/ keys = tags in original data, values = tags in original data
        :param with_tags: [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple tags, e.g. 'PER', 'LOC', ..
        :return: ner_tag_mapping: [dict] w/ keys = tags in original data, values = tags in formatted data
        """
        pass

    @abstractmethod
    def format_data(self, valid_fraction: float):
        """
        III: format data
        ----------------
        :param valid_fraction: [float]
        :return: -
        """
        pass

    ####################################################################################################################
    # BASE METHODS
    ####################################################################################################################
    def create_directory(self):
        """
        0: create directory for dataset
        -------------------------------
        :return: -
        """
        directory_path = f'{BASE_DIR}/datasets/ner/{self.ner_dataset}/analyze_data'
        os.makedirs(directory_path, exist_ok=True)

    def create_ner_tag_mapping(self, with_tags: bool, modify: bool):
        """
        II: create customized ner tag mapping to map tags in original data to tags in formatted data
        ----------------------------------------------------------------------------------------------
        :param with_tags:   [bool], if True: create tags with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                    if False: create simple tags, e.g. 'PER', 'LOC', ..
        :param modify:      [bool], if True: modify tags as specified in method modify_ner_tag_mapping()
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
            ner_tag_mapping = self.modify_ner_tag_mapping(ner_tag_mapping_original, with_tags=with_tags)
        else:
            ner_tag_mapping = ner_tag_mapping_original

        json_path = f'{BASE_DIR}/datasets/ner/{self.ner_dataset}/ner_tag_mapping.json'
        with open(json_path, 'w') as f:
            json.dump(ner_tag_mapping, f)

        print(f'> dumped the following dict to {json_path}:')
        print(ner_tag_mapping)

    def read_formatted_csv(self, phase):
        """
        III & IV: read formatted csv files
        ----------------------------------------------
        :param phase:         [str] 'train' or 'test'
        :return: num_sentences:    [int]
                 stats_aggregated: [pandas Series] with indices = tags, values = number of occurrences
        """
        file_path = f'{BASE_DIR}/datasets/ner/{self.ner_dataset}/{phase}.csv'

        columns = ['O'] + self.ner_tag_list

        try:
            df = pd.read_csv(file_path, sep='\t')
        except pd.io.common.EmptyDataError:
            df = None

        stats = pd.DataFrame([], columns=columns)

        if df is not None:
            tags = df.iloc[:, 0].apply(lambda x: x.split())

            for column in columns:
                stats[column] = tags.apply(lambda x: len([elem for elem in x if elem == column]))

            assert len(df) == len(stats)

        num_sentences = len(stats)
        stats_aggregated = stats.sum().to_frame().astype(int)
        stats_aggregated.columns = ['tags']

        return num_sentences, stats_aggregated

    def analyze_data(self):
        """
        IV: analyze data
        ----------------
        :created attr: stats_aggregated: [dict] w/ keys = 'total', 'train', 'valid', 'test' & values = [df]
        :return: -
        """
        log_path = f'datasets/ner/{self.ner_dataset}/analyze_data/{self.ner_dataset}.log'
        logger = get_file_logger(log_path)

        num_sentences = {'total': 0}
        self.stats_aggregated = {'total': None}
        phases = ['train', 'valid', 'test']

        for phase in phases:
            num_sentences[phase], _stats_aggregated_phase = self.read_formatted_csv(phase)
            num_sentences['total'] += num_sentences[phase]
            if self.stats_aggregated['total'] is None:
                self.stats_aggregated['total'] = _stats_aggregated_phase
            else:
                self.stats_aggregated['total'] = self.stats_aggregated['total'] + _stats_aggregated_phase

            self.stats_aggregated[phase] = \
                self._stats_aggregated_add_columns(_stats_aggregated_phase, num_sentences[phase])

        num_sentences_total = num_sentences['total']
        for phase in phases:
            logger.info('')
            logger.info(f'>>> {phase} <<<<')
            logger.info(f'num_sentences = {num_sentences[phase]} '
                        f'({100*num_sentences[phase]/num_sentences_total:.2f}% of total = {num_sentences_total}')
            logger.info('stats_aggregated:')
            logger.info(self.stats_aggregated[phase])

        self.stats_aggregated['total'] = self._stats_aggregated_add_columns(self.stats_aggregated['total'],
                                                                            num_sentences['total'])

        logger.info('')
        logger.info(f'>>> total <<<<')
        logger.info(f'num_sentences = {num_sentences}')
        logger.info('stats_aggregated:')
        logger.info(self.stats_aggregated['total'])

    def plot_data(self):
        fig_path = f'{self.dataset_path}/analyze_data/{self.ner_dataset}.png'
        Plots(self.stats_aggregated).plot(fig_path=fig_path)

    @staticmethod
    def _stats_aggregated_add_columns(df, number_of_sentences):
        """
        IV: analyze data
        ----------------
        :param df: ..
        :param number_of_sentences: ..
        :return: ..
        """
        df['sentences'] = int(number_of_sentences)
        df['tags/sentence'] = df['tags']/float(number_of_sentences)
        df['tags/sentence'] = df['tags/sentence'].apply(lambda x: '{:.2f}'.format(x))

        # relative tags w/ 0
        number_of_occurrences = df['tags'].sum()
        df['tags relative w/ 0'] = df['tags'] / number_of_occurrences
        df['tags relative w/ 0'] = df['tags relative w/ 0'].apply(lambda x: '{:.2f}'.format(x))

        # relative tags w/o 0
        number_of_filtered_occurrences = df['tags'].sum() - df.loc['O']['tags']
        df['tags relative w/o 0'] = df['tags'] / number_of_filtered_occurrences
        df['tags relative w/o 0']['O'] = 0
        df['tags relative w/o 0'] = df['tags relative w/o 0'].apply(lambda x: '{:.2f}'.format(x))

        return df
