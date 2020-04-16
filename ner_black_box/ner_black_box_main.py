
import os
from os.path import join
import mlflow
import shutil
from pkg_resources import Requirement
from pkg_resources import resource_filename, resource_isdir

from ner_black_box.utils.env_variable import env_variable


def main(
    flag,
    dataset=None,
    with_tags=False,
    modify=True,
    val_fraction=0.3,
    verbose=False,
    experiment_name=None,
    run_name=None,
    device='gpu',
    fp16=True,
):

    assert flag is not None, 'missing input flag (--init OR --set_up_dataset OR --analyze_data OR --run_experiment)'

    os.environ["MLFLOW_TRACKING_URI"] = env_variable('DIR_MLFLOW')

    ####################################################################################################################
    # --init
    ####################################################################################################################
    if flag == 'init':
        create_data_directory()

        for _dataset in ['conll2003', 'swedish_ner_corpus']:
            set_up_dataset(_dataset, with_tags, modify, val_fraction, verbose)

    ####################################################################################################################
    # --analyze_data
    ####################################################################################################################
    elif flag == 'analyze_data':
        assert dataset is not None, 'missing input --analyze_data --dataset <dataset>'
        analyze_data(dataset, verbose)

    ####################################################################################################################
    # --set_up_dataset
    ####################################################################################################################
    elif flag == 'set_up_dataset':
        assert dataset is not None, 'missing input --set_up_dataset --dataset <dataset>'
        set_up_dataset(dataset, with_tags, modify, val_fraction, verbose)

    ####################################################################################################################
    # --run_experiment
    ####################################################################################################################
    elif flag == 'run_experiment':
        assert experiment_name is not None, 'missing input --run_experiment --experiment_name <experiment_name>'
        run_experiment(experiment_name, run_name, device, fp16)


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


def analyze_data(_dataset, _verbose):
    _parameters = {
        'ner_dataset': _dataset,
        'verbose': False,
    }

    mlflow.projects.run(
        uri=join(os.environ.get('BASE_DIR'), 'ner_black_box'),
        entry_point='analyze_data',
        experiment_name=None,
        parameters=_parameters,
        use_conda=False,
    )


def set_up_dataset(_dataset, _with_tags, _modify, _val_fraction, _verbose):

    _parameters = {
        'ner_dataset': _dataset,
        'with_tags': _with_tags,
        'modify': _modify,
        'val_fraction': _val_fraction,
        'verbose': _verbose,
    }

    mlflow.projects.run(
        uri=join(os.environ.get('BASE_DIR'), 'ner_black_box'),
        entry_point='set_up_dataset',
        experiment_name=None,
        parameters=_parameters,
        use_conda=False,
    )


def run_experiment(_experiment_name, _run_name, _device, _fp16):
    _parameters = {
        'experiment_name': _experiment_name,
        'run_name': _run_name if _run_name else '',
        'device': _device,
        'fp16': _fp16,
    }

    mlflow.projects.run(
        uri=join(os.environ.get('BASE_DIR'), 'ner_black_box'),
        entry_point='run_experiment',
        experiment_name=_experiment_name,
        parameters=_parameters,
        use_conda=False,
    )
