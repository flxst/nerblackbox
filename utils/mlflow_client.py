
import os
import mlflow


class MLflowClient:

    def __init__(self):
        self.mlflow_artifact = 'mlruns/mlflow_artifact.txt'

        self._set_experiment_and_run_name()

    @staticmethod
    def _set_experiment_and_run_name():
        """
        set mlflow experiment and run name from env variables if available
        ------------------------------------------------------------------
        :return: -
        """
        try:
            experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        except KeyError:
            experiment_name = 'Default'

        mlflow.set_experiment(experiment_name)

        try:
            run_name = os.environ['MLFLOW_RUN_NAME']
        except KeyError:
            run_name = None

        print(experiment_name, run_name)
        if run_name:
            mlflow.start_run(run_name=run_name)

        print(f'mlflow experiment name: {experiment_name}')
        print(f'mlflow run        name: {run_name}')

    @staticmethod
    def log_params(_hyperparams):
        """
        mlflow hyperparameter logging
        -----------------------------
        :param _hyperparams:      [dict] for mlflow tracking
        :return:
        """
        # all hyperparameters
        mlflow.log_param('hyperparameters', _hyperparams)

        # most important hyperparameters
        most_important_hyperparameters = ['device', 'num_epochs', 'prune_ratio', 'lr_max', 'lr_schedule']
        for hyperparameter in most_important_hyperparameters:
            if hyperparameter.startswith('lr'):
                mlflow.log_param(hyperparameter, _hyperparams['learning_rate'][hyperparameter])
            else:
                mlflow.log_param(hyperparameter, _hyperparams[hyperparameter])


    @staticmethod
    def log_metrics(_epoch, _epoch_valid_metrics):
        """
        mlflow metrics logging
        -----------------------------
        :param: _epoch:               [int]
        :param: _epoch_valid_metrics  [dict] w/ keys 'loss', 'acc', 'f1' & values = [np array]
        :return: -
        """
        mlflow.log_metric('epoch', _epoch)
        for metric in ['loss', 'acc', 'f1_macro_all', 'f1_micro_all', 'f1_macro_fil', 'f1_micro_fil']:
            mlflow.log_metric(metric, _epoch_valid_metrics[metric])

    @staticmethod
    def log_time(_time):
        mlflow.log_metric('time', _time)

    def clear_artifact(self):
        """
        mlflow artifact logging
        -----------------------
        :return: -
        """
        with open(self.mlflow_artifact, "w") as f:
            f.write(' ')

    def log_artifact(self, content):
        """
        mlflow artifact logging
        -----------------------
        :param content: [str]
        :return: -
        """
        print(content)
        with open(self.mlflow_artifact, "a") as f:
            f.write(content + '\n')

    def finish(self):
        mlflow.log_artifact(self.mlflow_artifact)
        print(f'mlflow log artifact at {self.mlflow_artifact}')

        mlflow.end_run()

