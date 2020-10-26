import mlflow
from nerblackbox.modules.experiment_config.experiment_config import ExperimentConfig


class MLflowClient:
    def __init__(
        self, experiment_name, run_name, log_dirs, logged_metrics, default_logger
    ):
        """
        :param experiment_name: [str], e.g. 'Default'
        :param run_name:        [str], e.g. 'Default'
        :param log_dirs:        [Namespace], including 'mlflow_artifact' & 'default_logger_artifact'
        :param logged_metrics:  [list] of [str], e.g. ['all_precision_micro', 'all_precision_macro', ..]
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_dirs = log_dirs
        self.logged_metrics = logged_metrics  # TODO: not used !!
        self.default_logger = default_logger

    @staticmethod
    def log_params(params, hparams, experiment=False):
        """
        mlflow hyperparameter logging
        -----------------------------
        :param params:     [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        :param hparams:    [argparse.Namespace] attr: batch_size, max_seq_length, max_epochs, prune_ratio_*, lr_*
        :param experiment: [bool] whether run is part of an experiment w/ multiple runs
        :return:
        """
        if experiment:
            # log only run (hyper)parameters
            experiment_config = ExperimentConfig(
                experiment_name=params.experiment_name,
                run_name=params.run_name,
                device=params.device,
                fp16=params.fp16,
            )
            for k, v in experiment_config.get_params_and_hparams(
                run_name_nr=params.run_name_nr
            ).items():
                mlflow.log_param(k, v)
        else:
            # log hardcoded set of (hyper)parameters
            if params is not None:
                # all parameters
                mlflow.log_param("parameters", vars(params))

            if hparams is not None:
                # all hyperparameters
                mlflow.log_param("hyperparameters", vars(hparams))

                # most important hyperparameters
                most_important_hyperparameters = [
                    "prune_ratio_train",
                    "prune_ratio_val",
                    "prune_ratio_test",
                    "max_epochs",
                    "lr_max",
                    "lr_schedule",
                ]
                for hyperparameter in most_important_hyperparameters:
                    mlflow.log_param(hyperparameter, vars(hparams)[hyperparameter])

    def log_metric(self, _metric, _stopped_epoch):
        mlflow.log_metric(_metric, _stopped_epoch)

    def log_metrics(self, _epoch, _epoch_val_metrics):
        """
        mlflow metrics logging
        -----------------------------
        :param: _epoch:             [int]
        :param: _epoch_val_metrics  [dict] w/ keys 'loss', 'acc', 'f1' & values = [np array]
        :return: -
        """
        mlflow.log_metric("epoch", _epoch)
        for metric in _epoch_val_metrics.keys():
            _metric = metric.replace("[", "_").replace("]", "_").replace("+", "P")
            mlflow.log_metric(_metric, _epoch_val_metrics[metric])

    def log_classification_report(self, _classification_report, overwrite=False):
        """
        log classification report
        ------------------------------------------------------------------------------------
        :param: _classification_report: [str]
        :param: overwrite: [bool] if True, overwrite existing artifact, else append
        :return: -
        """
        if overwrite:
            self._clear_artifact()
        self._log_artifact(_classification_report)

    @staticmethod
    def log_time(_time):
        mlflow.log_metric("time", _time)

    def _clear_artifact(self):
        """
        mlflow artifact logging
        -----------------------
        :return: -
        """
        with open(self.log_dirs.mlflow_file, "w") as f:
            f.write(" ")

    def _log_artifact(self, content):
        """
        mlflow artifact logging
        -----------------------
        :param content: [str]
        :return: -
        """
        with open(self.log_dirs.mlflow_file, "a") as f:
            f.write(content + "\n")

    def finish_artifact_mlflow(self):
        # mlflow
        mlflow.log_artifact(self.log_dirs.mlflow_file)
        self.default_logger.log_debug(f"mlflow file at {self.log_dirs.mlflow_file}")

    def finish_artifact_logger(self):
        # default logger
        mlflow.log_artifact(self.log_dirs.log_file)
        self.default_logger.log_debug(f"log file at {self.log_dirs.log_file}")
