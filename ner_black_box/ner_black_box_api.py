

import os
from argparse import Namespace
from os.path import abspath, join

from ner_black_box.ner_black_box_main import NerBlackBoxMain


class NerBlackBoxApi:

    def __init__(self,
                 base_dir='.',
                 data_dir='data'):
        """
        :param base_dir: [str] relative path w.r.t. current directory
        :param data_dir: [str] relative path w.r.t. base_dir
        """

        os.environ['BASE_DIR'] = abspath(base_dir)
        os.environ['DATA_DIR'] = join(abspath(base_dir), data_dir)
        print('BASE_DIR = ', os.environ.get('BASE_DIR'))
        print('DATA_DIR = ', os.environ.get('DATA_DIR'))

    ####################################################################################################################
    # NER BLACK BOX
    ####################################################################################################################
    def init(self):
        nerbb = NerBlackBoxMain('init')
        nerbb.main()

    @staticmethod
    def analyze_data(dataset: str,
                     **kwargs_optional):
        """
        :param dataset:          [str] e.g. 'swedish_ner_corpus'
        :param kwargs_optional:
        :return: -
        """

        kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}
        kwargs['dataset'] = dataset

        nerbb = NerBlackBoxMain('analyze_data', **kwargs)
        nerbb.main()

    @staticmethod
    def set_up_dataset(dataset: str,
                       **kwargs_optional):
        """
        :param dataset:          [str] e.g. 'swedish_ner_corpus'
        :param kwargs_optional:
        :return: -
        """

        kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}
        kwargs['dataset'] = dataset

        nerbb = NerBlackBoxMain('set_up_dataset', **kwargs)
        nerbb.main()

    @staticmethod
    def show_experiment_config(experiment_name: str = None):
        kwargs = {'experiment_name': experiment_name}
        nerbb = NerBlackBoxMain('show_experiment_config', **kwargs)
        nerbb.main()

    @staticmethod
    def run_experiment(experiment_name: str = None,
                       **kwargs_optional):
        """
        :param         experiment_name: [str or None], e.g. exp0
        :return: -
        """

        kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}
        kwargs['experiment_name'] = experiment_name

        nerbb = NerBlackBoxMain('run_experiment', **kwargs)
        nerbb.main()

    @staticmethod
    def get_experiment_results(experiment_name: str = None):
        """
        :param         experiment_name: [str or None], e.g. exp0
        :return: -
        """
        nerbb = NerBlackBoxMain('get_experiment_results', experiment_name=experiment_name, usage='api')
        namespace = dict()
        namespace['experiment'], namespace['single_runs'], namespace['average_runs'], \
            namespace['best_single_run'], namespace['best_average_run'], namespace['best_model'] = nerbb.main()
        return Namespace(**namespace)

    @staticmethod
    def get_experiments(ids: tuple = (), as_df: bool = True):
        nerbb = NerBlackBoxMain('get_experiments', usage='api')
        return nerbb.main(ids, as_df)

    @staticmethod
    def get_experiments_results(ids: tuple = (), as_df: bool = True):
        nerbb = NerBlackBoxMain('get_experiments_results', usage='api')
        namespace = dict()
        namespace['best_single_runs'], namespace['best_average_runs'] = nerbb.main(ids, as_df)
        return Namespace(**namespace)
