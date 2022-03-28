import json
import os
from os.path import join, isdir, isfile

import omegaconf.errors
from transformers import AutoModelForTokenClassification
from omegaconf import DictConfig
from omegaconf.errors import ConfigKeyError
from transformers import AutoTokenizer
from nerblackbox.modules.ner_training.ner_model import NerModel

# TODO: this suppresses warnings (https://github.com/huggingface/transformers/issues/5421), root problem should be fixed
from transformers import logging

logging.set_verbosity_error()


class NerModelTrain2Model(NerModel):
    """
    class that translates (pytorch-lightning) NerModelTrain checkpoint to (transformers) NerModelPredict checkpoint
    """

    def __init__(self, hparams: DictConfig):
        """
        :param hparams: attr: experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__(hparams)

    def _preparations(self):
        self.annotation_classes = json.loads(self.hparams["annotation_classes"])
        self.pretrained_model_name = self.hparams["pretrained_model_name"]
        try:
            self.encoding = json.loads(self.hparams["encoding"])
            self.special_tokens = list(set(self.encoding.values()))
        except ConfigKeyError:  # TODO: get rid of this exception handling.
            print(
                "> Note: special tokens are loaded instead of encoding (model was trained with nerblackbox<=0.0.12)"
            )
            self.encoding = None
            self.special_tokens = json.loads(self.hparams["special_tokens"])
        self.max_seq_length = int(self.hparams["max_seq_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            do_lower_case=False,
            additional_special_tokens=self.special_tokens,
            use_fast=True,
        )  # do_lower_case needs to be False !!
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(self.annotation_classes),
            return_dict=False,
        )
        self.model.resize_token_embeddings(
            len(self.tokenizer)
        )  # due to additional_special_tokens

    ####################################################################################################################
    # EXPORT TO NER_MODEL_PROD
    ####################################################################################################################
    def export_to_ner_model_prod(self, checkpoint_path: str):
        assert checkpoint_path.endswith(".p") or checkpoint_path.endswith(
            ".ckpt"
        ), f"ERROR! checkpoint_path = {checkpoint_path} is no *.p or *.ckpt file."
        export_directory = checkpoint_path.strip(".p").strip(".ckpt")
        assert not isfile(
            export_directory
        ), f"ERROR! export_directory = {export_directory} seems to be a file instead of a directory"

        if not isdir(export_directory):
            os.makedirs(export_directory, exist_ok=False)

            # 1. max_seq_length
            path_max_seq_length = join(export_directory, "max_seq_length.json")
            with open(path_max_seq_length, "w") as f:
                json.dump(self.max_seq_length, f)

            # 2. annotation
            path_annotation_classes = join(export_directory, "annotation_classes.json")
            with open(path_annotation_classes, "w") as f:
                json.dump(self.annotation_classes, f)

            # 3. model
            self.model.save_pretrained(export_directory)

            # 4. tokenizer
            self.tokenizer.save_pretrained(export_directory)
