import os
from os.path import join
import json

from transformers import AutoTokenizer, AutoModelForTokenClassification

from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics
from nerblackbox.modules.ner_training.logging.mlflow_client import MLflowClient
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger
from nerblackbox.modules.ner_training.ner_model import NerModel


class NerModelTrain(NerModel):
    def __init__(self, hparams):
        """
        :param hparams: [argparse.Namespace] attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__(hparams)

    ####################################################################################################################
    # Abstract Base Methods ############################################################################################
    ####################################################################################################################
    def _preparations(self):
        """
        :created attr: default_logger    [DefaultLogger]
        :created attr: logged_metrics    [LoggedMetrics]
        :created attr: tokenizer         [transformers AutoTokenizer]
        :created attr: data_preprocessor [DataPreprocessor]
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]

        :created attr: mlflow_client          [MLflowClient]
        :created attr: epoch_metrics          [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: classification_reports [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: pretrained_model_name  [str]
        :created attr: dataloader             [dict] w/ keys 'train', 'val', 'test' & values = [torch Dataloader]
        :created attr: optimizer              [torch optimizer]
        :created attr: scheduler              [torch LambdaLR]

        :return: -
        """
        # train/val/test
        self._preparations_train()  # attr: default_logger, logged_metrics, mlflow_client, ..
        self._preparations_data_general()  # attr: tokenizer, data_preprocessor
        self._preparations_data_train()  # attr: tag_list, model, dataloader, optimizer, scheduler

    def _preparations_train(self):
        """
        :created attr: default_logger         [DefaultLogger]
        :created attr: logged_metrics         [LoggedMetrics]
        :created attr: mlflow_client          [MLflowClient]
        :created attr: epoch_metrics          [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: classification_reports [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: pretrained_model_name  [str]
        :return: -
        """
        self.default_logger = DefaultLogger(
            __file__, log_file=self.log_dirs.log_file, level=self.params.logging_level
        )

        self.logged_metrics = LoggedMetrics()

        self.mlflow_client = MLflowClient(
            experiment_name=self.params.experiment_name,
            run_name=self.params.run_name,
            log_dirs=self.log_dirs,
            logged_metrics=self.logged_metrics.as_flat_list(),
            default_logger=self.default_logger,
        )
        self.mlflow_client.log_params(
            self.params, self.hparams, experiment=self.experiment
        )

        self.epoch_metrics = {"val": dict(), "test": dict()}
        self.classification_reports = {"val": dict(), "test": dict()}

        try:
            # use transformers model
            AutoTokenizer.from_pretrained(self.params.pretrained_model_name)
            self.pretrained_model_name = self.params.pretrained_model_name
        except ValueError:
            # use local model
            self.pretrained_model_name = join(
                os.environ.get("DATA_DIR"),
                "pretrained_models",
                self.params.pretrained_model_name,
            )

    def _preparations_data_train(self):
        """
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]
        :created attr: dataloader        [dict] w/ keys 'train', 'val', 'test' & values = [torch Dataloader]
        :created attr: optimizer         [torch optimizer]
        :created attr: scheduler         [torch LambdaLR]
        :return: -
        """
        # input_examples & tag_list
        input_examples, self.tag_list = self.data_preprocessor.get_input_examples_train(
            dataset_name=self.params.dataset_name,
            prune_ratio={
                "train": self.params.prune_ratio_train,
                "val": self.params.prune_ratio_val,
                "test": self.params.prune_ratio_test,
            },
        )
        self.default_logger.log_debug("> self.tag_list:", self.tag_list)
        self.hparams.tag_list = json.dumps(
            self.tag_list
        )  # save for PREDICT (see below)

        # model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_name, num_labels=len(self.tag_list)
        )

        # dataloader
        self.dataloader = self.data_preprocessor.to_dataloader(
            input_examples, self.tag_list, batch_size=self._hparams.batch_size
        )

        # optimizer
        self.optimizer = self._create_optimizer(
            self._hparams.lr_max, fp16=self.params.fp16
        )

        # learning rate
        self.scheduler = self._create_scheduler(
            self._hparams.lr_warmup_epochs,
            self._hparams.lr_schedule,
            self._hparams.lr_num_cycles,
        )
