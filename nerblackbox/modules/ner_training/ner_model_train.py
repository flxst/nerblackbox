import json

from transformers import AutoModelForTokenClassification
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics
from nerblackbox.modules.ner_training.logging.mlflow_client import MLflowClient
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger
from nerblackbox.modules.ner_training.ner_model import NerModel
from nerblackbox.modules.ner_training.annotation_tags.input_examples_utils import (
    InputExamplesUtils,
)
from nerblackbox.modules.utils.util_functions import read_special_tokens


class NerModelTrain(NerModel):
    def __init__(self, hparams: DictConfig):
        """
        :param hparams: attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
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
        :created attr: special_tokens    [list] of [str] e.g. ["[NEWLINE]", "[TAB]"]
        :created attr: data_preprocessor [DataPreprocessor]
        :created attr: annotation        [Annotation]
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
        self._preparations_data_train()  # attr: annotation, model, dataloader, optimizer, scheduler

    def _preparations_train(self):
        """
        :created attr: default_logger         [DefaultLogger]
        :created attr: logged_metrics         [LoggedMetrics]
        :created attr: mlflow_client          [MLflowClient]
        :created attr: epoch_metrics          [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: classification_reports [dict] w/ keys = 'val', 'test' & values = [dict]
        :created attr: special_tokens         [list] of [str] e.g. ["[NEWLINE]", "[TAB]"]
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

        self.special_tokens = read_special_tokens(self.params.dataset_name)
        self.default_logger.log_info(f"> read special tokens: {self.special_tokens}")

    def _preparations_data_train(self):
        """
        :created attr: annotation        [Annotation]
        :created attr: model             [transformers AutoModelForTokenClassification]
        :created attr: dataloader        [dict] w/ keys 'train', 'val', 'test' & values = [torch Dataloader]
        :created attr: optimizer         [torch optimizer]
        :created attr: scheduler         [torch LambdaLR]
        :return: -
        """
        # input_examples & annotation
        (
            input_examples,
            self.annotation,
        ) = self.data_preprocessor.get_input_examples_train(
            prune_ratio={
                "train": self.params.prune_ratio_train,
                "val": self.params.prune_ratio_val,
                "test": self.params.prune_ratio_test,
            },
            dataset_name=self.params.dataset_name,
            train_on_val=self.params.train_on_val,
            train_on_test=self.params.train_on_test,
        )
        self.default_logger.log_info(
            f"> annotation scheme found: {self.annotation.scheme}"
        )
        if self.params.annotation_scheme == "auto":
            self.params.annotation_scheme = self.annotation.scheme
        elif self.params.annotation_scheme != self.annotation.scheme:
            # convert annotation_classes
            input_examples = InputExamplesUtils.convert_annotation_scheme(
                input_examples=input_examples,
                annotation_scheme_source=self.annotation.scheme,
                annotation_scheme_target=self.params.annotation_scheme,
            )
            self.annotation = self.annotation.change_scheme(
                new_scheme=self.params.annotation_scheme
            )
            self.default_logger.log_info(
                f"> annotation scheme converted to {self.params.annotation_scheme}"
            )

        self.default_logger.log_debug(
            "> self.annotation.classes:", self.annotation.classes
        )

        self.hparams.annotation_classes = json.dumps(
            self.annotation.classes
        )  # save for NerModelPredict
        self.hparams.special_tokens = json.dumps(
            self.special_tokens
        )  # save for NerModelPredict

        # model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(self.annotation.classes),
            return_dict=False,
        )
        self.model.resize_token_embeddings(
            len(self.tokenizer)
        )  # due to additional_special_tokens

        # dataloader
        self.dataloader = self.data_preprocessor.to_dataloader(
            input_examples, self.annotation.classes, batch_size=self._hparams.batch_size
        )

        # optimizer
        self.optimizer: Optimizer = self._create_optimizer(
            self._hparams.lr_max, fp16=self.params.fp16
        )

        # learning rate
        self.scheduler: LambdaLR = self._create_scheduler(
            self._hparams.lr_warmup_epochs,
            self._hparams.lr_schedule,
            self.hyperparameters.max_epochs,
            self._hparams.lr_num_cycles,
        )
