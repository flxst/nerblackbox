
import pandas as pd
from mlflow.tracking import MlflowClient

from os.path import abspath, dirname, join
BASE_DIR = abspath(dirname(__file__))

from ner_black_box.ner_training.lightning_ner_model import LightningNerModel
from ner_black_box.scripts import run_experiment as run_experiment_python_script


class NerBlackBoxApi:

    def __init__(self,
                 path_results='results',
                 path_experiment_configs='experiment_configs'):

        self.path_results = join(BASE_DIR, path_results)
        self.path_experiment_configs = path_experiment_configs

        self.client = MlflowClient(join(self.path_results, 'mlruns'))
        self._get_experiments()

        self.experiment_name = None
        self.experiment = None
        self.runs = None
        self.best_run = None
        self.best_model = None

    def _get_experiments(self):
        """
        :created attr: experiment_id2name [dict] w/ keys = experiment_id [str] & values = experiment_name [str]
        :created attr: experiment_name2id [dict] w/ keys = experiment_name [str] & values = experiment_id [str]
        :return: -
        """
        self.experiment_id2name = {elem['_experiment_id']: elem['_name']
                                   for elem in [vars(experiment) for experiment in self.client.list_experiments()]
                                   }
        self.experiment_name2id = {v: k for k, v in self.experiment_id2name.items()}

    ####################################################################################################################
    # ONE EXPERIMENT
    ####################################################################################################################
    def set_experiment(self, experiment_name: str):
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
        if experiment_name is None and self.experiment_name is None:
            print('experiment_name needs to be set')
            return None
        elif experiment_name is not None and self.experiment_name != experiment_name:
            self.experiment_name = experiment_name
        print(f'> experiment_name = {self.experiment_name}')

    def show_experiment_config(self, experiment_name: str = None):
        """
        print experiment config
        -----------------------
        :param         experiment_name: [str or None], e.g. exp0
        :return: -
        """
        self._check_experiment_name(experiment_name)

        path_experiment_config = join(BASE_DIR, self.path_experiment_configs, f'{self.experiment_name}.ini')
        with open(path_experiment_config, 'r') as file:
            lines = file.read()
        print()
        print(lines)

    def run_experiment(self, experiment_name: str = None):
        """
        :param         experiment_name: [str or None], e.g. exp0
        :return: -
        """
        self._check_experiment_name(experiment_name)

        run_experiment_python_script.main(self.experiment_name)

        self._get_experiments()  # needs to updated to get results from experiment that was run

    def get_experiment_results(self, experiment_name: str = None):
        """
        :param         experiment_name: [str or None], e.g. exp0
        :changed attr: experiment:      [pandas DataFrame] overview on experiment parameters
        :changed attr: runs:            [pandas DataFrame] overview on run parameters & results
        :changed attr: best_run:        [dict] overview on best run parameters & results
        :changed attr: best_model:      [LightningNerModel]
        :return: -
        """
        self._check_experiment_name(experiment_name)

        if self.experiment_name not in self.experiment_name2id.keys():
            print(f'no experiment with experiment_name = {self.experiment_name} found')
            print(f'experiments that were found:')
            print(self.experiment_name2id)
            return None

        experiment_id = self.experiment_name2id[self.experiment_name]
        self.experiment, self.runs, self.best_run = self._get_single_experiment_results(experiment_id)

        self.best_model = LightningNerModel.load_from_checkpoint(checkpoint_path=self.best_run['checkpoint'])
        self.best_model.eval()

    ####################################################################################################################
    # ONE EXPERIMENT: HELPER
    ####################################################################################################################
    def _get_single_experiment_results(self, experiment_id: str):
        """
        :param experiment_id: [str], e.g. '0'
        :return: _experiment: [pandas DataFrame] overview on experiment parameters
        :return: _runs:       [pandas DataFrame] overview on run parameters & results
        :return: _best_run:   [dict] overview on best run parameters & results
        """
        experiment_name = self.experiment_id2name[experiment_id]
        runs = self.client.search_runs(experiment_id)

        _experiment, _runs = self._parse_and_create_dataframe(runs)

        _df_best_run = _runs.iloc[0, :]
        best_run_id = _df_best_run[('info', 'run_id')]
        best_run_name = _df_best_run[('info', 'run_name')]
        best_run_epoch_best = _df_best_run[('metrics', 'epoch_best')]
        best_run_chk_f1_micro = _df_best_run[('metrics', 'chk_f1_micro')]

        _best_run = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'run_id': best_run_id,
            'run_name': best_run_name,
            'chk_f1_micro': best_run_chk_f1_micro,
            'checkpoint': join(self.path_results,
                               'checkpoints',
                               experiment_name,
                               best_run_name,
                               f'_ckpt_epoch_{best_run_epoch_best}.ckpt')
        }
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

        return pd.DataFrame(experiments_overview) if as_df else experiments_overview

    def get_best_runs_of_multiple_experiments(self, ids: tuple = (), as_df: bool = False):
        """
        :param ids:   [tuple] of [str], e.g. ('4', '5')
        :param as_df: [bool] if True, return [pandas DataFrame], else return [dict]
        :return: best_runs_overview [pandas DataFrame] or [dict]
        """
        experiments_filtered = self._filter_experiments_by_ids(ids)

        best_runs_overview = list()
        for _id in sorted(list(experiments_filtered.keys())):
            _, _, best_run = self._get_single_experiment_results(_id)
            best_runs_overview.append(best_run)

        return pd.DataFrame(best_runs_overview) if as_df else best_runs_overview

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
        fields_metrics = ['epoch_best', 'epoch_stopped', 'allP_loss', 'chk_f1_micro']
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
        _runs = pd.DataFrame(parameters_runs).sort_values(by=('metrics', 'chk_f1_micro'), ascending=False)
        return _experiment, _runs

    ####################################################################################################################
    # ADDITONAL HELPER
    ####################################################################################################################
    @staticmethod
    def show_as_df(_dict):
        return pd.DataFrame(_dict)
