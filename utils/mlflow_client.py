
import mlflow


class MLflowClient:

    def __init__(self, experiment_name, run_name, log_dir, logged_metrics):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mlflow_artifact = f'{log_dir}/mlflow_artifact.txt'
        self.logged_metrics = logged_metrics
        print(self.experiment_name, self.run_name)

        mlflow.set_tracking_uri(log_dir)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

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
        most_important_hyperparameters = ['max_epochs',
                                          'prune_ratio_train',
                                          'prune_ratio_valid',
                                          'lr_max',
                                          'lr_schedule']
        for hyperparameter in most_important_hyperparameters:
            mlflow.log_param(hyperparameter, _hyperparams[hyperparameter])

    def log_metrics(self, _epoch, _epoch_valid_metrics):
        """
        mlflow metrics logging
        -----------------------------
        :param: _epoch:               [int]
        :param: _epoch_valid_metrics  [dict] w/ keys 'loss', 'acc', 'f1' & values = [np array]
        :return: -
        """
        mlflow.log_metric('epoch', _epoch)
        for metric in _epoch_valid_metrics.keys():
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

    def finish_artifact(self):
        mlflow.log_artifact(self.mlflow_artifact)
        print(f'mlflow log artifact at {self.mlflow_artifact}')

    def finish(self):
        self.finish_artifact()
        mlflow.end_run()
