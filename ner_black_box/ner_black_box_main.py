
import os
from os.path import join
import mlflow
import shutil
from pkg_resources import Requirement
from pkg_resources import resource_filename, resource_isdir
import pandas as pd
from mlflow.tracking import MlflowClient

from ner_black_box.utils.env_variable import env_variable
from ner_black_box.utils.util_functions import epoch2checkpoint
from ner_black_box.utils.util_functions import get_run_name, compute_mean_and_dmean


class NerBlackBoxMain:

    def __init__(self,
                 flag,
                 dataset_name=None,
                 with_tags=False,
                 modify=True,
                 val_fraction=0.3,
                 verbose=False,
                 experiment_name=None,
                 run_name=None,
                 device='gpu',
                 fp16=False,
                 usage='cli',
                 ):
        assert flag is not None, 'missing input flag (--init OR --set_up_dataset OR --analyze_data OR --run_experiment)'

        os.environ["MLFLOW_TRACKING_URI"] = env_variable('DIR_MLFLOW')

        self.flag = flag
        self.dataset_name = dataset_name
        self.with_tags = with_tags
        self.modify = modify
        self.val_fraction = val_fraction
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.device = device
        self.fp16 = fp16
        self.usage = usage

        if os.path.isdir(os.environ.get('DATA_DIR')):
            self._set_client_and_get_experiments()
        else:
            # will be set in init() method
            self.client = None
            self.experiment_id2name = None
            self.experiment_name2id = None

        self.experiment = None
        self.single_runs = None
        self.average_runs = None
        self.best_single_run = None
        self.best_average_run = None
        self.best_model = None

    def main(self, ids=(), as_df=True):

        ################################################################################################################
        # --init
        ################################################################################################################
        if self.flag == 'init':
            self.create_data_directory()

            for _dataset_name in ['conll2003', 'swedish_ner_corpus']:
                self.set_up_dataset(_dataset_name, self.with_tags, self.modify, self.val_fraction, self.verbose)

            self._set_client_and_get_experiments()

        ################################################################################################################
        # --analyze_data
        ################################################################################################################
        elif self.flag == 'analyze_data':
            assert self.dataset_name is not None, 'missing input --analyze_data <dataset_name>'
            self.analyze_data(self.dataset_name, self.verbose)

        ################################################################################################################
        # --set_up_dataset
        ################################################################################################################
        elif self.flag == 'set_up_dataset':
            assert self.dataset_name is not None, 'missing input --set_up_dataset <dataset_name>'
            self.set_up_dataset(self.dataset_name, self.with_tags, self.modify, self.val_fraction, self.verbose)

        ################################################################################################################
        # --show_experiment_config
        ################################################################################################################
        elif self.flag == 'show_experiment_config':
            assert self.experiment_name is not None, 'missing input --show_experiment_config <experiment_name>'
            self.show_experiment_config(self.experiment_name)

        ################################################################################################################
        # --run_experiment
        ################################################################################################################
        elif self.flag == 'run_experiment':
            assert self.experiment_name is not None, 'missing input --run_experiment <experiment_name>'
            self.run_experiment(self.experiment_name, self.run_name, self.device, self.fp16)

        ################################################################################################################
        # --get_experiment_results
        ################################################################################################################
        elif self.flag == 'get_experiment_results':
            assert self.experiment_name is not None, 'missing input --get_experiment_results <experiment_name>'
            assert self.usage in ['cli', 'api'], 'missing usage'
            return self.get_experiment_results(self.experiment_name)

        ################################################################################################################
        # --get_experiments
        ################################################################################################################
        elif self.flag == 'get_experiments':
            assert self.usage in ['cli', 'api'], 'missing usage'
            return self.get_experiments(ids, as_df)

        ################################################################################################################
        # --get_experiments_best_runs
        ################################################################################################################
        elif self.flag == 'get_experiments_results':
            assert self.usage in ['cli', 'api'], 'missing usage'
            return self.get_experiments_results(ids, as_df)

    @staticmethod
    def create_data_directory():
        if resource_isdir(Requirement.parse('ner_black_box'), 'ner_black_box/data'):
            data_source = resource_filename(Requirement.parse('ner_black_box'), 'ner_black_box/data')
            data_target = os.environ.get('DATA_DIR')
            if os.path.isdir(data_target):
                print(f'--init: target {data_target} already exists')
            else:
                shutil.copytree(data_source, data_target)
                print(f'--init: target {data_target} created')
        else:
            print('--init not executed successfully')
            exit(0)

    @staticmethod
    def analyze_data(_dataset_name, _verbose):
        _parameters = {
            'ner_dataset': _dataset_name,
            'verbose': _verbose,
        }

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse('ner_black_box'), 'ner_black_box'),
            entry_point='analyze_data',
            experiment_name='analyze_data',
            parameters=_parameters,
            use_conda=False,
        )

    @staticmethod
    def set_up_dataset(_dataset_name, _with_tags, _modify, _val_fraction, _verbose):

        _parameters = {
            'ner_dataset': _dataset_name,
            'with_tags': _with_tags,
            'modify': _modify,
            'val_fraction': _val_fraction,
            'verbose': _verbose,
        }

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse('ner_black_box'), 'ner_black_box'),
            entry_point='set_up_dataset',
            experiment_name='set_up_dataset',
            parameters=_parameters,
            use_conda=False,
        )

    def run_experiment(self, _experiment_name, _run_name, _device, _fp16):
        _parameters = {
            'experiment_name': _experiment_name,
            'run_name': _run_name if _run_name else '',
            'device': _device,
            'fp16': int(_fp16),
        }

        mlflow.projects.run(
            uri=resource_filename(Requirement.parse('ner_black_box'), 'ner_black_box'),
            entry_point='run_experiment',
            experiment_name=_experiment_name,
            parameters=_parameters,
            use_conda=False,
        )

        self._get_experiments()  # needs to updated to get results from experiment that was run
        self.get_experiment_results(_experiment_name)

    def _set_client_and_get_experiments(self):
        """
        :created attr: client             [Mlflow client]
        :created attr: experiment_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        :created attr: experiment_name2id [dict] w/ keys = experiment_name [str] & values = experiment_id [str]
        :return: -
        """
        self.client = MlflowClient()
        self._get_experiments()

    def _get_experiments(self):
        """
        :created attr: experiment_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        :created attr: experiment_name2id [dict] w/ keys = experiment_name [str] & values = experiment_id [str]
        :return: -
        """
        self.experiment_id2name = {elem['_experiment_id']: elem['_name']
                                   for elem in [vars(experiment) for experiment in self.client.list_experiments()]
                                   if elem['_name'] != 'Default'
                                   }
        self.experiment_name2id = {v: k for k, v in self.experiment_id2name.items()}

    ####################################################################################################################
    # SINGLE EXPERIMENT
    ####################################################################################################################
    def set_experiment(self,
                       experiment_name: str):
        """
        :param         experiment_name: [str], e.g. exp0
        :changed attr: experiment_name: [str], e.g. exp0
        :return: -
        """
        self.experiment_name = experiment_name

    def _check_experiment_name(self, experiment_name):
        """
        :param         experiment_name: [str or None], e.g. exp0
        :changed attr: experiment_name: [str],         e.g. exp0
        :return: -
        """
        from ner_black_box.utils.env_variable import env_variable

        if experiment_name is None and self.experiment_name is None:
            print('experiment_name needs to be set')
            return None
        elif experiment_name is not None and self.experiment_name != experiment_name:
            self.experiment_name = experiment_name
        print(f'> experiment_name = {self.experiment_name}')

        self.path_experiment_config = join(env_variable('DIR_EXPERIMENT_CONFIGS'), f'{self.experiment_name}.ini')
        print(f'> experiment_config = {self.path_experiment_config}')

    def show_experiment_config(self, experiment_name: str = None):
        """
        print experiment config
        -----------------------
        :param         experiment_name: [str or None], e.g. exp0
        :return: -
        """
        self._check_experiment_name(experiment_name)

        with open(self.path_experiment_config, 'r') as file:
            lines = file.read()
        print()
        print(lines)

    def get_experiment_results(self, experiment_name: str = None):
        """
        :param         experiment_name: [str or None], e.g. 'exp0'
        :changed attr: experiment:      [pandas DataFrame] overview on experiment parameters
        :changed attr: runs:            [pandas DataFrame] overview on run parameters & results
        :changed attr: best_run:        [dict] overview on best run parameters & results
        :changed attr: best_model:      [LightningNerModel]
        :return: -
        """
        from ner_black_box.ner_training.lightning_ner_model_predict import LightningNerModelPredict

        if experiment_name not in self.experiment_name2id.keys():
            print(f'no experiment with experiment_name = {self.experiment_name} found')
            print(f'experiments that were found:')
            print(self.experiment_name2id)
            return None, None, None, None

        experiment_id = self.experiment_name2id[self.experiment_name]
        self.experiment, self.single_runs, self.average_runs, self.best_single_run, self.best_average_run = \
            self._get_single_experiment_results(experiment_id)

        if self.usage == 'cli':
            print('### single runs ###')
            print(self.single_runs)
            print()
            print('### average runs ###')
            print(self.average_runs)
        else:
            if self.best_single_run['checkpoint'] is not None:
                self.best_model = LightningNerModelPredict.load_from_checkpoint(self.best_single_run['checkpoint'])
            else:
                self.best_model = None

            return self.experiment, self.single_runs, self.average_runs, \
                self.best_single_run, self.best_average_run, self.best_model

    ####################################################################################################################
    # SINGLE EXPERIMENT: HELPER
    ####################################################################################################################
    def _get_single_experiment_results(self, experiment_id: str, verbose: bool = False):
        """
        :param experiment_id: [str], e.g. '0'
        :return: _experiment:       [pandas DataFrame] overview on experiment parameters
        :return: _single_runs:      [pandas DataFrame] overview on run parameters & single  results
        :return: _average_runs:     [pandas DataFrame] overview on run parameters & average results
        :return: _best_single_run:  [dict] overview on best run parameters & single  results
        :return: _best_average_run: [dict] overview on best run parameters & average results
        """
        experiment_name = self.experiment_id2name[experiment_id]
        runs = self.client.search_runs(experiment_id)

        _experiment, _single_runs, _average_runs = self._parse_and_create_dataframe(runs, verbose=verbose)

        # best run
        if _experiment is not None and _single_runs is not None:
            _df_best_single_run = _single_runs.iloc[0, :]
            best_single_run_id = _df_best_single_run[('info', 'run_id')]
            best_single_run_name_nr = _df_best_single_run[('info', 'run_name_nr')]
            best_single_run_epoch_best = _df_best_single_run[('metrics', 'epoch_best')]
            best_single_run_epoch_best_val_chk_f1_micro = _df_best_single_run[('metrics', 'epoch_best_val_chk_f1_micro')]
            best_single_run_epoch_best_test_chk_f1_micro = _df_best_single_run[('metrics', 'epoch_best_test_chk_f1_micro')]

            checkpoint = join(
                env_variable('DIR_CHECKPOINTS'),
                experiment_name,
                best_single_run_name_nr,
                epoch2checkpoint(best_single_run_epoch_best),
            )

            _best_single_run = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'run_id': best_single_run_id,
                'run_name_nr': best_single_run_name_nr,
                'epoch_best_val_chk_f1_micro': best_single_run_epoch_best_val_chk_f1_micro,
                'epoch_best_test_chk_f1_micro': best_single_run_epoch_best_test_chk_f1_micro,
                'checkpoint': checkpoint if os.path.isfile(checkpoint) else None,
            }
        else:
            _best_single_run = None

        # best run average
        if _experiment is not None and _average_runs is not None:
            _df_best_average_run = _average_runs.iloc[0, :]
            best_average_run_name = _df_best_average_run[('info', 'run_name')]
            best_average_run_epoch_best_val_chk_f1_micro = _df_best_average_run[('metrics', 'epoch_best_val_chk_f1_micro')]
            d_best_average_run_epoch_best_val_chk_f1_micro = _df_best_average_run[('metrics', 'd_epoch_best_val_chk_f1_micro')]
            best_average_run_epoch_best_test_chk_f1_micro = _df_best_average_run[('metrics', 'epoch_best_test_chk_f1_micro')]
            d_best_average_run_epoch_best_test_chk_f1_micro = _df_best_average_run[('metrics', 'd_epoch_best_test_chk_f1_micro')]

            _best_average_run = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'run_name': best_average_run_name,
                'epoch_best_val_chk_f1_micro': best_average_run_epoch_best_val_chk_f1_micro,
                'd_epoch_best_val_chk_f1_micro': d_best_average_run_epoch_best_val_chk_f1_micro,
                'epoch_best_test_chk_f1_micro': best_average_run_epoch_best_test_chk_f1_micro,
                'd_epoch_best_test_chk_f1_micro': d_best_average_run_epoch_best_test_chk_f1_micro,
            }
        else:
            _best_average_run = None
        return _experiment, _single_runs, _average_runs, _best_single_run, _best_average_run

    ####################################################################################################################
    # ALL EXPERIMENTS
    ####################################################################################################################
    def get_experiments(self, ids: tuple = (), as_df: bool = False):
        """
        :param ids:   [tuple] of [str], e.g. ('4', '5')
        :param as_df: [bool] if True, return [pandas DataFrame], else return [dict]
        :return: experiments_overview [pandas DataFrame] or [dict]
        """
        experiments_id2name_filtered = self._filter_experiments_by_ids(ids)

        experiments_overview = sorted(
            [{'experiment_id': k, 'experiment_name': v} for k, v in experiments_id2name_filtered.items()],
            key=lambda elem: elem['experiment_id']
        )
        df = pd.DataFrame(experiments_overview) if as_df else experiments_overview
        if self.usage == 'cli':
            print('### experiments ###')
            print(df)
        else:
            return df

    def get_experiments_results(self, ids: tuple = (), as_df: bool = False, verbose: bool = False):
        """
        :param ids:     [tuple] of [str], e.g. ('4', '5')
        :param as_df:   [bool] if True, return [pandas DataFrame], else return [dict]
        :param verbose: [bool]
        :return: best_runs_overview [pandas DataFrame] or [dict]
        """
        experiments_filtered = self._filter_experiments_by_ids(ids)

        best_single_runs_overview = list()
        best_average_runs_overview = list()
        for _id in sorted(list(experiments_filtered.keys())):
            _, _, _, best_single_run, best_average_run = self._get_single_experiment_results(_id, verbose=verbose)
            if best_single_run:
                best_single_runs_overview.append(best_single_run)
            if best_average_run:
                best_average_runs_overview.append(best_average_run)

        df_single = pd.DataFrame(best_single_runs_overview) if as_df else best_single_runs_overview
        df_average = pd.DataFrame(best_average_runs_overview) if as_df else best_average_runs_overview
        if self.usage == 'cli':
            print('### best single runs ###')
            print(df_single)
            print()
            print('### best average runs ###')
            print(df_average)
        else:
            return df_single, df_average

    ####################################################################################################################
    # ALL EXPERIMENTS: HELPER
    ####################################################################################################################
    def _filter_experiments_by_ids(self, _ids):
        """
        get _experiments_id2name [dict] with _ids as keys
        -------------------------------------------------
        :param _ids:  [tuple] of [str], e.g. ('4', '5')
        :return: _experiments_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        """
        if len(_ids) == 0:
            _experiments_id2name = self.experiment_id2name
        else:
            _experiments_id2name = {k: v
                                    for k, v in self.experiment_id2name.items()
                                    if k in _ids}
        return _experiments_id2name

    @staticmethod
    def _parse_and_create_dataframe(_runs, verbose=False):
        """
        turn mlflow Run objects (= search_runs() results) into data frames
        ------------------------------------------------------------------
        :param _runs:   [list] of [mlflow.entities.Run objects]
        :param verbose: [bool]
        :return: _experiment:   [pandas DataFrame] overview on experiment parameters
        :return: _single_runs:  [pandas DataFrame] overview on single  run parameters & results
        :return: _average_runs: [pandas DataFrame] overview on average run parameters & results
        """
        fields_metrics = ['epoch_best', 'epoch_stopped', 'epoch_best_val_chk_f1_micro', 'epoch_best_test_chk_f1_micro']

        ###########################################
        # parameters_experiment & parameters_runs
        ###########################################
        parameters_runs = dict()
        for i in range(len(_runs)):
            if len(_runs[i].data.metrics) == 0:  # experiment
                parameters_experiment = {k: [v] for k, v in _runs[i].data.params.items()}
            else:  # run
                if ('info', 'run_id') not in parameters_runs.keys():
                    parameters_runs[('info', 'run_id')] = [_runs[i].info.run_id]
                else:
                    parameters_runs[('info', 'run_id')].append(_runs[i].info.run_id)

                if ('info', 'run_name_nr') not in parameters_runs.keys():
                    parameters_runs[('info', 'run_name_nr')] = [_runs[i].data.tags['mlflow.runName']]
                else:
                    parameters_runs[('info', 'run_name_nr')].append(_runs[i].data.tags['mlflow.runName'])

                for k, v in _runs[i].data.params.items():
                    if ('params', k) not in parameters_runs.keys():
                        parameters_runs[('params', k)] = [v]
                    else:
                        parameters_runs[('params', k)].append(v)

                for k in fields_metrics:
                    if ('metrics', k) not in parameters_runs.keys():
                        try:
                            parameters_runs[('metrics', k)] = [_runs[i].data.metrics[k]]
                        except:
                            parameters_runs[('metrics', k)] = [-1]
                    else:
                        try:
                            parameters_runs[('metrics', k)].append(_runs[i].data.metrics[k])
                        except:
                            parameters_runs[('metrics', k)] = [-1]

        _experiment = pd.DataFrame(parameters_experiment, index=['experiment']).T
        for k in ['epoch_best', 'epoch_stopped']:
            try:
                parameters_runs[('metrics', k)] = [int(elem) for elem in parameters_runs[('metrics', k)]]
            except:
                parameters_runs[('metrics', k)] = [-1]

        ###########################################
        # parameters_runs_average
        ###########################################
        def average(_parameters_runs):
            _parameters_runs_average = {('info', 'run_name'): list()}
            _parameters_runs_average.update({
                k: list()
                for k in _parameters_runs.keys()
                if k[0] not in ['info', 'metrics']
            })
            _parameters_runs_average.update({
                k: list()
                for k in [
                    ('metrics', 'epoch_best_val_chk_f1_micro'),
                    ('metrics', 'd_epoch_best_val_chk_f1_micro'),
                    ('metrics', 'epoch_best_test_chk_f1_micro'),
                    ('metrics', 'd_epoch_best_test_chk_f1_micro'),
                ]
            })

            runs_name_nr = _parameters_runs[('info', 'run_name_nr')]
            nr_runs = len(runs_name_nr)
            runs_name = list(set([get_run_name(run_name_nr) for run_name_nr in runs_name_nr]))
            indices = {
                run_name: [idx for idx in range(nr_runs) if get_run_name(runs_name_nr[idx]) == run_name]
                for run_name in runs_name
            }

            def get_mean_and_dmean(_parameters_runs, phase):
                values = [_parameters_runs[('metrics', f'epoch_best_{phase}_chk_f1_micro')][idx]
                          for idx in indices[run_name]]
                return compute_mean_and_dmean(values)

            #######################
            # loop over runs_name
            #######################
            for run_name in runs_name:
                # add ('info', 'run_name')
                _parameters_runs_average[('info', 'run_name')].append(run_name)

                # add ('params', *)
                random_index = indices[run_name][0]
                keys_kept = [k for k in _parameters_runs.keys() if k[0] == 'params']
                for key in keys_kept:
                    _parameters_runs_average[key].append(_parameters_runs[key][random_index])

                # add ('metrics', *)
                val_mean, val_dmean = get_mean_and_dmean(_parameters_runs, 'val')
                test_mean, test_dmean = get_mean_and_dmean(_parameters_runs, 'test')
                metrics = {
                    'epoch_best_val_chk_f1_micro': val_mean,
                    'd_epoch_best_val_chk_f1_micro': val_dmean,
                    'epoch_best_test_chk_f1_micro': test_mean,
                    'd_epoch_best_test_chk_f1_micro': test_dmean,
                }
                for k in metrics.keys():
                    key = ('metrics', k)
                    _parameters_runs_average[key].append(metrics[k])

            return _parameters_runs_average

        ###########################################
        # sort & return
        ###########################################
        by = ('metrics', 'epoch_best_val_chk_f1_micro')
        try:
            _single_runs = pd.DataFrame(parameters_runs).sort_values(by=by, ascending=False)
        except:
            _single_runs = None

        try:
            parameters_runs_average = average(parameters_runs)
            _average_runs = pd.DataFrame(parameters_runs_average).sort_values(by=by, ascending=False)
        except:
            _average_runs = None

        return _experiment, _single_runs, _average_runs

    ####################################################################################################################
    # ADDITONAL HELPER
    ####################################################################################################################
    @staticmethod
    def show_as_df(_dict):
        return pd.DataFrame(_dict)
