
import os
from os.path import join
import mlflow
import shutil
from pkg_resources import Requirement
from pkg_resources import resource_filename, resource_isdir
import pandas as pd
from mlflow.tracking import MlflowClient

from ner_black_box.utils.env_variable import env_variable


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
                 fp16=True,
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
        self.runs = None
        self.best_run = None
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
        elif self.flag == 'get_experiments_best_runs':
            assert self.usage in ['cli', 'api'], 'missing usage'
            return self.get_experiments_best_runs(ids, as_df)

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
            'verbose': False,
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
            'fp16': _fp16,
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
        self.experiment, self.runs, self.best_run = self._get_single_experiment_results(experiment_id)

        if self.usage == 'cli':
            print(self.runs)
        else:
            if self.best_run['checkpoint'] is not None:
                self.best_model = LightningNerModelPredict.load_from_checkpoint(self.best_run['checkpoint'])
                self.best_model.eval()
                self.best_model.freeze()
            else:
                self.best_model = None

            return self.experiment, self.runs, self.best_run, self.best_model

    ####################################################################################################################
    # SINGLE EXPERIMENT: HELPER
    ####################################################################################################################
    def _get_single_experiment_results(self, experiment_id: str, verbose: bool = False):
        """
        :param experiment_id: [str], e.g. '0'
        :return: _experiment: [pandas DataFrame] overview on experiment parameters
        :return: _runs:       [pandas DataFrame] overview on run parameters & results
        :return: _best_run:   [dict] overview on best run parameters & results
        """
        experiment_name = self.experiment_id2name[experiment_id]
        runs = self.client.search_runs(experiment_id)

        _experiment, _runs = self._parse_and_create_dataframe(runs, verbose=verbose)

        if _experiment is not None and _runs is not None:
            _df_best_run = _runs.iloc[0, :]
            best_run_id = _df_best_run[('info', 'run_id')]
            best_run_name = _df_best_run[('info', 'run_name')]
            best_run_epoch_best = _df_best_run[('metrics', 'epoch_best')]
            best_run_epoch_best_val_chk_f1_micro = _df_best_run[('metrics', 'epoch_best_val_chk_f1_micro')]
            best_run_epoch_best_test_chk_f1_micro = _df_best_run[('metrics', 'epoch_best_test_chk_f1_micro')]

            checkpoint = join(
                env_variable('DIR_CHECKPOINTS'),
                experiment_name,
                best_run_name,
                f'_ckpt_epoch_{best_run_epoch_best}.ckpt',
            )

            _best_run = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'run_id': best_run_id,
                'run_name': best_run_name,
                'epoch_best_val_chk_f1_micro': best_run_epoch_best_val_chk_f1_micro,
                'epoch_best_test_chk_f1_micro': best_run_epoch_best_test_chk_f1_micro,
                'checkpoint': checkpoint if os.path.isfile(checkpoint) else None,
            }
        else:
            _best_run = None
        return _experiment, _runs, _best_run

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
            print(df)
        else:
            return df

    def get_experiments_best_runs(self, ids: tuple = (), as_df: bool = False, verbose: bool = False):
        """
        :param ids:   [tuple] of [str], e.g. ('4', '5')
        :param as_df: [bool] if True, return [pandas DataFrame], else return [dict]
        :return: best_runs_overview [pandas DataFrame] or [dict]
        """
        experiments_filtered = self._filter_experiments_by_ids(ids)

        best_runs_overview = list()
        for _id in sorted(list(experiments_filtered.keys())):
            _, _, best_run = self._get_single_experiment_results(_id, verbose=verbose)
            if best_run:
                best_runs_overview.append(best_run)

        df = pd.DataFrame(best_runs_overview) if as_df else best_runs_overview
        if self.usage == 'cli':
            print(df)
        else:
            return df

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
        :return: _experiment: [pandas DataFrame] overview on experiment parameters
        :return: _runs:       [pandas DataFrame] overview on run parameters & results
        """
        fields_metrics = ['epoch_best', 'epoch_stopped', 'epoch_best_val_chk_f1_micro', 'epoch_best_test_chk_f1_micro']
        parameters_runs = dict()

        for i in range(len(_runs)):
            if len(_runs[i].data.metrics) == 0:  # experiment
                parameters_experiment = {k: [v] for k, v in _runs[i].data.params.items()}
            else:  # run
                if ('info', 'run_id') not in parameters_runs.keys():
                    parameters_runs[('info', 'run_id')] = [_runs[i].info.run_id]
                else:
                    parameters_runs[('info', 'run_id')].append(_runs[i].info.run_id)

                if ('info', 'run_name') not in parameters_runs.keys():
                    parameters_runs[('info', 'run_name')] = [_runs[i].data.tags['mlflow.runName']]
                else:
                    parameters_runs[('info', 'run_name')].append(_runs[i].data.tags['mlflow.runName'])

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

        for k in ['epoch_best', 'epoch_stopped']:
            try:
                parameters_runs[('metrics', k)] = [int(elem) for elem in parameters_runs[('metrics', k)]]
            except:
                parameters_runs[('metrics', k)] = [-1]

        if verbose:
            print()
            print('parameters_experiment:', parameters_experiment)
            print()
            print('parameters_runs:', parameters_runs)
        _experiment = pd.DataFrame(parameters_experiment, index=['experiment']).T
        try:
            _runs = pd.DataFrame(parameters_runs).sort_values(by=('metrics', 'epoch_best_val_chk_f1_micro'),
                                                              ascending=False)
            return _experiment, _runs
        except:
            return None, None

    ####################################################################################################################
    # ADDITONAL HELPER
    ####################################################################################################################
    @staticmethod
    def show_as_df(_dict):
        return pd.DataFrame(_dict)
