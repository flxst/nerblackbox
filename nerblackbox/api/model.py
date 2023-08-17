import json
import string
from os.path import join, isdir, isfile
from typing import Tuple, Any, Optional
import numpy as np
import copy

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List, Union, Dict
from torch.nn.functional import softmax
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from itertools import product

from nerblackbox.modules.ner_training.annotation_tags.token_tags import TokenTags
from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.api.store import Store
from nerblackbox.api.dataset import Dataset
from nerblackbox.modules.ner_training.metrics.ner_metrics import NerMetrics
from nerblackbox.modules.training_results import TrainingResults
from nerblackbox.modules.ner_training.ner_model_train2model import (
    NerModelTrain2Model,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.csv_reader import (
    CsvReader,
)
from nerblackbox.modules.ner_training.annotation_tags.tags import Tags
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter

PREDICTIONS = List[List[Dict[str, Any]]]
EVALUATION_DICT = Dict[str, Dict[str, Dict[str, Optional[float]]]]

VERBOSE = False
DEBUG = False


class Model:
    r"""
    model that predicts tags for given input text
    """

    @classmethod
    def from_training(cls, training_name: str) -> Optional["Model"]:
        r"""load best model from training.

        Args:
            training_name: name of the training, e.g. "training0"

        Returns:
            model: best model from training
        """
        training_exists, training_results = Store.get_training_results_single(
            training_name
        )
        assert (
            training_exists
        ), f"ERROR! training = {training_name} does not exist."

        assert isinstance(
            training_results, TrainingResults
        ), f"ERROR! training_results expected to be an instance of TrainingResults."

        if training_results.best_single_run is None:
            print(
                f"> ATTENTION! could not find results for training = {training_name}"
            )
            return None
        elif (
            "checkpoint" not in training_results.best_single_run.keys()
            or training_results.best_single_run["checkpoint"] is None
        ):
            print(
                f"> ATTENTION! there is no checkpoint for training = {training_name}."
            )
            return None
        else:
            checkpoint_path_train = training_results.best_single_run["checkpoint"]
            checkpoint_path_predict = checkpoint_path_train.strip(".ckpt")

            # translate NerModelTrain checkpoint to Model checkpoint if necessary
            if not Model.checkpoint_exists(checkpoint_path_predict):
                ner_model_train2model = NerModelTrain2Model.load_from_checkpoint(
                    checkpoint_path_train
                )
                ner_model_train2model.export_to_ner_model_prod(checkpoint_path_train)

            return Model(checkpoint_path_predict)

    @classmethod
    def from_checkpoint(cls, checkpoint_directory: str) -> Optional["Model"]:
        r"""

        Args:
            checkpoint_directory: path to the checkpoint directory

        Returns:
            model: best model from training
        """
        if not cls.checkpoint_exists(checkpoint_directory):
            print(
                f"> ATTENTION! could not find checkpoint directory at {checkpoint_directory}"
            )
            return None
        else:
            return Model(checkpoint_directory)

    @classmethod
    def from_huggingface(cls, repo_id: str, dataset: Optional[str] = None) -> Optional["Model"]:
        r"""

        Args:
            repo_id: id of the huggingface hub repo id, e.g. 'KB/bert-base-swedish-cased-ner'
            dataset: should be provided in case model is missing information on id2label in config

        Returns:
            model: model
        """
        # download files and get local file paths in cache directory
        filenames = [
            "config.json",
            "pytorch_model.bin",
            "vocab.txt",
            "vocab.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "sentencepiece.bpe.model",
        ]
        local_file_paths = []
        count_vocab = 4
        for filename in filenames:
            try:
                local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
                local_file_paths.append(local_file_path)
            except EntryNotFoundError:
                if filename in ["special_tokens_map.json", "tokenizer_config.json", "merges.txt"]:
                    # some models do not have these files
                    pass
                elif filename in ["vocab.txt", "vocab.json", "sentencepiece.bpe.model", "tokenizer.json"]:
                    # one of them needs to exist
                    count_vocab -= 1
                else:
                    raise Exception(f"ERROR! could not find filename = {filename} - cannot use model.")
        assert count_vocab > 0, f"ERROR! found no tokenizer files, should be at least one."

        # extract cache directory
        cache_directories = [
            "/".join(local_file_path.split("/")[:-1])
            for local_file_path in local_file_paths
        ]
        assert (
            len(set(cache_directories)) == 1
        ), f"ERROR! cache directory could not be found due to inconsistency."
        cache_directory = cache_directories[0]

        # create Model from files in cache directory
        return Model(cache_directory, dataset=dataset)

    @classmethod
    def checkpoint_exists(cls, checkpoint_directory: str) -> bool:
        return isdir(checkpoint_directory)

    def __init__(
        self,
        checkpoint_directory: str,
        batch_size: int = 16,
        max_seq_length: Optional[int] = None,
        dataset: Optional[str] = None,
    ):
        r"""
        Args:
            checkpoint_directory: path to the checkpoint directory
            batch_size: batch size used by dataloader
            max_seq_length: maximum sequence length (Optional). Loaded from checkpoint if not specified.
            dataset: should be provided in case model is missing information on id2label in config
        """
        # 0. device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1: config (max_seq_length & annotation)
        path_config = join(checkpoint_directory, "config.json")
        with open(path_config, "r") as f:
            config = json.load(f)

        if max_seq_length:
            self.max_seq_length = max_seq_length
        elif "max_seq_length" in config.keys():
            self.max_seq_length = config["max_seq_length"]
        else:
            self.max_seq_length = config["max_position_embeddings"]

        if "id2label" not in config.keys():
            raise Exception("ERROR! config.json does not contain id2label - cannot use model.")

        if not list(config["id2label"].values())[0].startswith("LABEL"):
            # most cases, as it should be
            id2label = {int(_id): label for _id, label in config["id2label"].items()}
        elif not list(config["label2id"].keys())[0].startswith("LABEL"):
            # e.g. IIC/bert-base-spanish-wwm-cased-ehealth_kd
            id2label = {int(_id): label for label, _id in config["label2id"].items()}
        elif dataset is not None:
            # get information from dataset
            try:
                formatter = AutoFormatter.for_dataset(
                    dataset
                )
                id2label = {int(_id): label for _id, label in config["id2label"].items()}
                labels = formatter.tags
                assert len(labels) == len(id2label), f"ERROR!"
                id2label = {i: label for i, label in enumerate(labels)}
            except Exception:
                raise Exception(
                    f"config.json does not contain proper id2label and dataset cannot be parsed. cannot use model."
                )
        else:
            # e.g. malduwais/distilbert-base-uncased-finetuned-ner
            # e.g. gagan3012/bert-tiny-finetuned-ner
            raise Exception(
                f"ERROR! config.json does not contain proper id2label - cannot use model. try to provide dataset."
            )

        id2label = dict(sorted(id2label.items()))
        label2id = {label: int(_id) for _id, label in id2label.items()}

        self.annotation_classes = list(id2label.values())
        self.annotation_scheme = derive_annotation_scheme(id2label)

        # 3. model
        self.model = AutoModelForTokenClassification.from_pretrained(
            checkpoint_directory,
            id2label=id2label,
            label2id=label2id,
            return_dict=False,
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        # 4. tokenizer
        path_config = join(checkpoint_directory, "tokenizer_config.json")
        if isfile(path_config):
            with open(path_config, "r") as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = dict()
        self._analyze_tokenizer(tokenizer_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_directory,
            add_prefix_space=True,  # only used for SentencePiece tokenizers (needed for pretokenized text)
        )

        # 5. batch_size (dataloader)
        self.batch_size = batch_size

    def predict_on_file(self, input_file: str, output_file: str) -> None:
        r"""
        predict tags for all input texts in input file, write results to output file

        Args:
            input_file: e.g. strangnas/test.jsonl
            output_file: e.g. strangnas/test_anonymized.jsonl
        """
        print(f"> read   input_file = {input_file}")
        with open(input_file, "r") as f:
            input_lines = [json.loads(line) for line in f]

        output_lines = list()
        for input_line in input_lines:
            text = input_line["text"]
            tags = self.predict(text)[0]
            output_line = {
                "text": text,
                "tags": tags,
            }
            output_lines.append(output_line)

        print(f"> write output_file = {output_file}")
        with open(output_file, "w") as f:
            for line in output_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def predict(
        self,
        input_texts: Union[str, List[str]],
        level: str = "entity",
        autocorrect: bool = False,
        is_pretokenized: bool = False,
    ) -> PREDICTIONS:
        r"""predict tags for input texts. output on entity or word level.

        Examples:
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="word", autocorrect=False)
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "I-ORG"},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
            #     {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-LOC"},
            # ]]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="word", autocorrect=True)
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
            #     {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-LOC"},
            # ]]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="entity", autocorrect=False)
            # [[
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "LOC"},
            # ]]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="entity", autocorrect=True)
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "LOC"},
            # ]]
            ```

        Args:
            input_texts:   e.g. ["example 1", "example 2"]
            level:         "entity" or "word"
            autocorrect:   if True, autocorrect annotation scheme (e.g. B- and I- tags).
            is_pretokenized: True if input_texts are pretokenized

        Returns:
            predictions: [list] of predictions for the different examples.
                         each list contains a [list] of [dict] w/ keys = char_start, char_end, word, tag
        """
        return self._predict(
            input_texts,
            level,
            autocorrect,
            proba=False,
            is_pretokenized=is_pretokenized,
        )

    def predict_proba(
        self, input_texts: Union[str, List[str]], is_pretokenized: bool = False
    ) -> PREDICTIONS:
        r"""predict probability distributions for input texts. output on word level.

        Examples:
            ```
            predict_proba(["arbetsförmedlingen finns i stockholm"])
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "proba_dist: {"O": 0.21, "B-ORG": 0.56, ..}},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "proba_dist: {"O": 0.87, "B-ORG": 0.02, ..}},
            #     {"char_start": "25", "char_end": "26", "token": "i", "proba_dist: {"O": 0.95, "B-ORG": 0.01, ..}},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "proba_dist: {"O": 0.14, "B-ORG": 0.22, ..}},
            # ]]
            ```

        Args:
            input_texts:   e.g. ["example 1", "example 2"]
            is_pretokenized: True if input_texts are pretokenized

        Returns:
            predictions: [list] of probability predictions for different examples.
                         each list contains a [list] of [dict] w/ keys = char_start, char_end, word, proba_dist
                         where proba_dist = [dict] that maps self.annotation.classes to probabilities
        """
        return self._predict(
            input_texts,
            level="word",
            autocorrect=False,
            proba=True,
            is_pretokenized=is_pretokenized,
        )

    def _predict(
        self,
        input_texts: Union[str, List[str]],
        level: str = "entity",
        autocorrect: bool = False,
        proba: bool = False,
        is_pretokenized: bool = False,
    ) -> PREDICTIONS:
        r"""predict tags or probabilities for tags

        Args:
            input_texts:  e.g. ["example 1", "example 2"]
            level:        "entity" or "word"
            autocorrect:  if True, autocorrect annotation scheme (e.g. B- and I- tags).
            proba:        if True, predict probabilities instead of labels (on word level)
            is_pretokenized: True if input_texts are pretokenized

        Returns:
            predictions: [list] of [list] of [dict] w/ keys = char_start, char_end, word, tag/proba_dist
                         where proba_dist = [dict] that maps self.annotation.classes to probabilities
        """
        # --- check input arguments ---
        assert level in [
            "entity",
            "word",
        ], f"ERROR! model prediction level = {level} unknown, needs to be entity or word."
        if proba:
            assert level == "word" and autocorrect is False, (
                f"ERROR! probability predictions require level = word and autocorrect = False. "
                f"level = {level} and autocorrect = {autocorrect} not possible."
            )
        # ------------------------------

        # 0. ensure input_texts is a list
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        number_of_input_texts = len(input_texts)

        # 1. pure model inference
        self.data_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            do_lower_case=self.tokenizer.do_lower_case if hasattr(self.tokenizer, "do_lower_case") else False,
            max_seq_length=self.max_seq_length,
            default_logger=PseudoDefaultLogger(),
        )
        (
            input_examples,
            input_texts_pretokenized,
            pretokenization_offsets,
        ) = self.data_preprocessor.get_input_examples_predict(
            examples=input_texts,
            is_pretokenized=is_pretokenized,
        )

        dataloader_all, offsets_all = self.data_preprocessor.to_dataloader(
            input_examples, self.annotation_classes, batch_size=self.batch_size
        )
        dataloader = dataloader_all["predict"]
        offsets = offsets_all["predict"]

        inputs_batches = list()
        outputs_batches = list()

        for sample in dataloader:
            inputs_batches.append(sample["input_ids"])
            sample.pop("labels")
            sample = {k: v.to(self.device) for k, v in sample.items()}
            with torch.no_grad():
                outputs_batch = self.model(**sample)[
                    0
                ]  # shape = [batch_size, seq_length, num_labels]
            outputs_batches.append(outputs_batch.detach().cpu())

        inputs = torch.cat(inputs_batches)  # tensor of shape = [batch_size, seq_length]
        outputs = torch.cat(
            outputs_batches
        )  # tensor of shape = [batch_size, seq_length, num_labels]

        ################################################################################################################
        ################################################################################################################
        tokens: List[List[str]] = [
            self.tokenizer.convert_ids_to_tokens(input_ids)
            for input_ids in inputs.tolist()
        ]  # List[List[str]] with len = batch_size

        predictions: List[List[Any]]
        if proba:
            predictions = turn_tensors_into_tag_probability_distributions(
                annotation_classes=self.annotation_classes,
                outputs=outputs,
            )  # List[List[Dict[str, float]]] with len = batch_size
        else:
            predictions_ids = torch.argmax(
                outputs, dim=2
            )  # tensor of shape [batch_size, seq_length]
            predictions = [
                [
                    self.model.config.id2label[prediction]
                    for prediction in predictions_ids[i].numpy()
                ]
                for i in range(len(predictions_ids))
            ]  # List[List[str]] with len = batch_size

        # merge
        tokens = [
            merge_slices_for_single_document(tokens[offsets[i]: offsets[i + 1]])
            for i in range(len(offsets) - 1)
        ]  # List[List[str]] with len = number_of_input_texts
        predictions = [
            merge_slices_for_single_document(predictions[offsets[i]: offsets[i + 1]])
            for i in range(len(offsets) - 1)
        ]  # List[List[str]] or List[List[Dict[str, float]]] with len = number_of_input_texts

        assert len(tokens) == len(
            predictions
        ), f"ERROR! len(tokens) = {len(tokens)} should equal len(predictions) = {len(predictions)}"
        assert (
            len(tokens) == number_of_input_texts
        ), f"ERROR! len(tokens) = {len(tokens)} should equal len(input_texts) = {number_of_input_texts}"
        assert (
            len(predictions) == number_of_input_texts
        ), f"ERROR! len(predictions) = {len(predictions)} should equal len(input_texts) = {number_of_input_texts}"

        # 2. post processing
        predictions = [
            self._post_processing(
                level,
                autocorrect,
                proba,
                input_texts[i],
                input_texts_pretokenized[i],
                pretokenization_offsets[i]
                if pretokenization_offsets is not None
                else None,
                tokens[i],
                predictions[i],
            )
            for i in range(number_of_input_texts)
        ]

        return predictions

    def _post_processing(
        self,
        level: str,
        autocorrect: bool,
        proba: bool,
        input_text: str,
        input_text_pretokenized: str,
        pretokenization_offsets: Optional[List[Tuple[int, int]]],
        tokens: List[str],
        predictions: List[Any],
    ) -> List[List[Dict[str, str]]]:
        r"""
        Args:
            level: "word" or "entity"
            autocorrect: e.g. False
            proba: e.g. False
            input_text: e.g. "we are in stockholm."
            input_text_pretokenized: e.g. "we are in stockholm ."
            pretokenization_offsets: e.g. [(0,2), (3,6), (7,9), (10,19), (19,20)]
            tokens: e.g. ["we", "are", "in", "stockholm", "."]
            predictions: e.g. ["O", "O", "O", "B-LOC", "O"]

        Returns:
            input_text_word_predictions: e.g. [
                {'char_start': '0', 'char_end': '2', 'token': 'we', 'tag': 'O'},
                {'char_start': '3', 'char_end': '6', 'token': 'are', 'tag': 'O'},
                {'char_start': '7', 'char_end': '9', 'token': 'in', 'tag': 'O'},
                {'char_start': '10', 'char_end': '19', 'token': 'stockholm', 'tag': 'B-LOC'},
                {'char_start': '20', 'char_end': '21', 'token': '.', 'tag': 'O'},
            ]
        """
        ######################################
        # 1 input_text, merged chunks, tokens -> words
        ######################################
        _token_predictions: List[
            Tuple[str, Union[str, Dict[str, float]]]
        ] = merge_subtoken_to_token_predictions(tokens, predictions, self.tokenizer_special, self.tokenizer_type)

        token_predictions: List[Dict[str, Union[str, Dict]]] = restore_unknown_tokens(
            _token_predictions, input_text_pretokenized, verbose=VERBOSE
        )

        if autocorrect or level == "entity":
            assert (
                proba is False
            ), f"ERROR! autocorrect = {autocorrect} / level = {level} not allowed if proba = {proba}"

        #######################################
        token_predictions_str: List[Dict[str, str]] = assert_typing(token_predictions)
        token_tags = TokenTags(token_predictions_str, scheme=self.annotation_scheme)
        token_tags.merge_tokens_to_words()
        #######################################
        if pretokenization_offsets is not None:
            token_tags.unpretokenize(pretokenization_offsets)

        if autocorrect:
            token_tags.restore_annotation_scheme_consistency()

        if level == "entity":
            token_tags.merge_tokens_to_entities(
                original_text=input_text, verbose=VERBOSE
            )

        token_tags.correct_sentence_piece_tokens()

        predictions = token_tags.as_list()

        return predictions

    def evaluate_on_dataset(
        self,
        dataset_name: str,
        dataset_format: str = "infer",
        phase: str = "test",
        class_mapping: Optional[Dict[str, str]] = None,
        number: Optional[int] = None,
        derived_from_jsonl: bool = False,
        rounded_decimals: Optional[int] = 3,
    ) -> EVALUATION_DICT:
        r"""
        evaluate model on dataset from huggingface or local dataset in jsonl or csv format

        Args:
            dataset_name: e.g. 'conll2003'
            dataset_format: 'huggingface', 'jsonl', 'csv'
            phase: e.g. 'test'
            class_mapping: e.g. {"PER": "PI", "ORG": "PI}
            number: e.g. 100
            derived_from_jsonl:
            rounded_decimals: if not None, results will be rounded to provided decimals

        Returns:
            evaluation_dict:
                Dict with keys [label][level][metric]
                where label in ['micro', 'macro'],
                level in ['entity', 'token']
                metric in ['precision', 'recall', 'f1', 'precision_seqeval', 'recall_seqeval', 'f1_seqeval']
                and values = float between 0 and 1

        """
        dataset_formats = ["infer", "jsonl", "csv", "huggingface"]
        phases = ["train", "val", "test"]
        assert (
            dataset_format in dataset_formats
        ), f"ERROR! dataset_format={dataset_format} unknown (known={dataset_formats})"
        assert phase in phases, f"ERROR! phase = {phase} unknown (known={phases})"

        store_path = Store.get_path()
        assert isinstance(
            store_path, str
        ), f"ERROR! type(store_path) = {type(store_path)} shoudl be str."

        if dataset_format == "infer":
            file_path_jsonl = join(
                store_path, "datasets", dataset_name, f"{phase}.jsonl"
            )
            file_path_csv = join(store_path, "datasets", dataset_name, f"{phase}.csv")
            if isfile(file_path_jsonl):
                dataset_format = "jsonl"
            elif isfile(file_path_csv):
                dataset_format = "csv"
            else:
                dataset_format = "huggingface"
            print(f"> dataset_format = {dataset_format} (inferred)")

        dir_path = join(store_path, "datasets", dataset_name)
        if dataset_format == "huggingface":
            evaluation_dict = self._evaluate_on_huggingface(
                dataset_name, phase, class_mapping, number
            )
        elif dataset_format == "jsonl":
            evaluation_dict = self._evaluate_on_jsonl(
                dir_path, phase, class_mapping, number
            )
        elif dataset_format == "csv":
            evaluation_dict = self._evaluate_on_csv(
                dir_path, phase, class_mapping, number, derived_from_jsonl
            )
        else:
            raise Exception(f"ERROR! dataset_format = {dataset_format} unknown.")

        if rounded_decimals is None:
            return evaluation_dict
        else:
            return round_evaluation_dict(evaluation_dict, rounded_decimals)

    def _evaluate_on_huggingface(
        self,
        dataset_name: str,
        phase: str,
        class_mapping: Optional[Dict[str, str]] = None,
        number: Optional[int] = None,
    ) -> EVALUATION_DICT:
        r"""
        evaluate model on dataset from huggingface

        Args:
            dataset_name: e.g. 'conll2003'
            phase: e.g. 'test'
            class_mapping: e.g. {"PER": "PI", "ORG": "PI}
            number: e.g. 100

        Returns:
            evaluation_dict:
                Dict with keys [label][level][metric]
                where label in ['micro', 'macro'],
                level in ['entity', 'token']
                metric in ['precision', 'recall', 'f1', 'precision_seqeval', 'recall_seqeval', 'f1_seqeval']
                and values = float between 0 and 1
        """
        dataset = Dataset(name=dataset_name, source="HF")
        dataset.set_up()
        store_path = Store.get_path()
        dir_path = f"{store_path}/datasets/{dataset_name}"

        file_path_jsonl = join(
            store_path, "datasets", dataset_name, f"{phase}.jsonl"
        )
        file_path_csv = join(store_path, "datasets", dataset_name, f"{phase}.csv")
        if isfile(file_path_jsonl):
            return self._evaluate_on_jsonl(
                dir_path, phase, class_mapping, number
            )
        elif isfile(file_path_csv):
            return self._evaluate_on_csv(
                dir_path,
                phase=phase,
                class_mapping=class_mapping,
                number=number,
                derived_from_jsonl=False,
            )
        else:
            raise Exception(f"ERROR! evaluation on HF dataset failed, neither jsonl nor csv files seem to exist.")

    def _evaluate_on_jsonl(
        self,
        dir_path: str,
        phase: str,
        class_mapping: Optional[Dict[str, str]] = None,
        number: Optional[int] = None,
    ) -> EVALUATION_DICT:
        r"""
        evaluate model on local dataset in jsonl format

        Args:
            dir_path: e.g. './store/datasets/my_dataset'
            phase: e.g. 'test'
            class_mapping: e.g. {"PER": "PI", "ORG": "PI"}
            number: e.g. 100

        Returns:
            evaluation_dict:
                Dict with keys [label][level][metric]
                where label in ['micro', 'macro'],
                level in ['entity', 'token']
                metric in ['precision', 'recall', 'f1', 'precision_seqeval', 'recall_seqeval', 'f1_seqeval']
                and values = float between 0 and 1
        """
        assert isdir(dir_path), f"ERROR! could not find {dir_path}"
        data_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            do_lower_case=False,
            max_seq_length=self.max_seq_length,
            default_logger=PseudoDefaultLogger(),
        )
        data_preprocessor._pretokenize(dir_path)

        return self._evaluate_on_csv(
            dir_path,
            phase=phase,
            class_mapping=class_mapping,
            number=number,
            derived_from_jsonl=True,
        )

    def _evaluate_on_csv(
        self,
        dir_path: str,
        phase: str,
        class_mapping: Optional[Dict[str, str]] = None,
        number: Optional[int] = None,
        derived_from_jsonl: bool = False,
    ) -> EVALUATION_DICT:
        r"""
        evaluate model on local dataset in csv format

        Args:
            dir_path: e.g. './store/datasets/my_dataset'
            phase: e.g. 'test'
            class_mapping: e.g. {"PER": "PI", "ORG": "PI}
            number: e.g. 100
            derived_from_jsonl: should be True is csv was created from jsonl through pretokenization

        Returns:
            evaluation_dict:
                Dict with keys [label][level][metric]
                where label in ['micro', 'macro'],
                level in ['entity', 'token']
                metric in ['precision', 'recall', 'f1', 'precision_seqeval', 'recall_seqeval', 'f1_seqeval']
                and values = float between 0 and 1
        """
        # derived_from_jsonl = True  => pretokenized_<phase>.csv is used
        # derived_from_jsonl = False => <phase>.csv is used
        file_path = join(
            dir_path,
            f"pretokenized_{phase}.csv" if derived_from_jsonl else f"{phase}.csv",
        )
        assert isfile(file_path), f"ERROR! could not find {file_path}"

        csv_reader = CsvReader(
            dir_path,
            pretokenized=derived_from_jsonl is False,
            do_lower_case=self.tokenizer.do_lower_case if hasattr(self.tokenizer, "do_lower_case") else False,
            default_logger=None,
        )
        data = csv_reader.get_input_examples(phase)
        ground_truth = [elem.tags.split() for elem in data]
        input_texts = [elem.text for elem in data]
        if number is not None:
            ground_truth = ground_truth[:number]
            input_texts = input_texts[:number]

        _predictions = self.predict(input_texts, level="word", is_pretokenized=True)
        predictions = [
            [elem["tag"] for elem in _prediction] for _prediction in _predictions
        ]

        def convert_plain_to_bio(_predictions: List[str]) -> List[str]:
            tags = Tags(_predictions)
            return tags.convert_scheme("plain", "bio")

        if self.annotation_scheme == "plain":
            print(
                "> ATTENTION! predictions converted from plain to bio annotation scheme!"
            )
            predictions = [
                convert_plain_to_bio(prediction) for prediction in predictions
            ]

        # check that ground truth and predictions have same lengths
        assert len(ground_truth) == len(
            predictions
        ), f"ERROR! #ground_truth = {len(ground_truth)}, #predictions = {len(predictions)}"
        for i in range(len(ground_truth)):
            assert len(ground_truth[i]) == len(predictions[i]), (
                f"ERROR! #ground_truth[{i}] = {len(ground_truth[i])} ({ground_truth[i]}), "
                f"#predictions[{i}] = {len(predictions[i])} ({predictions[i]}),"
                f"input_texts[{i}] = {input_texts[i]}"
            )

        return self._evaluate(ground_truth, predictions, class_mapping)

    @staticmethod
    def _evaluate(
        ground_truth: List[List[str]],
        predictions: List[List[str]],
        class_mapping: Optional[Dict[str, str]] = None,
    ) -> EVALUATION_DICT:
        r"""
        evaluate by comparing ground_truth with predictions (after applying class_mapping)

        Args:
            ground_truth: e.g. [["B-PER", "I-PER"], ["O", "B-ORG"]]
            predictions: e.g. [["B-PER", "O"], ["O", "B-ORG"]]
            class_mapping: e.g. {"PER": "PI", "ORG": "PI}

        Returns:
            evaluation_dict:
                Dict with keys [label][level][metric]
                where label in ['micro', 'macro'],
                level in ['entity', 'token']
                metric in ['precision', 'recall', 'f1', 'precision_seqeval', 'recall_seqeval', 'f1_seqeval']
                and values = float between 0 and 1
        """
        # class mapping
        def map_class(_class: str) -> str:
            r"""
            maps class according to class_mapping
            Args:
                _class: e.g. "B-PER"

            Returns:
                _class_new: e.g. "B-PI"

            """
            assert isinstance(
                class_mapping, dict
            ), f"ERROR! type(class_mapping) = {type(class_mapping)} should be dict"
            if _class == "O":
                return "O"
            else:
                assert (
                    "-" in _class
                ), f"ERROR! class = {_class} expected to be of BIO scheme."
                _class_prefix = _class.split("-")[0]
                _class_plain = _class.split("-")[-1]
                if _class_plain in class_mapping.keys():
                    _class_plain_new = class_mapping[_class_plain]
                    _class_new = f"{_class_prefix}-{_class_plain_new}"
                else:
                    _class_new = "O"
                return _class_new

        if class_mapping is not None:
            predictions = [
                [map_class(elem) for elem in sublist] for sublist in predictions
            ]

        # flatten
        true_flat = [elem for sublist in ground_truth for elem in sublist]
        pred_flat = [elem for sublist in predictions for elem in sublist]
        assert len(true_flat) == len(
            pred_flat
        ), f"ERROR! true_flat = {len(true_flat)}, #pred_flat = {len(pred_flat)}"

        # 4. evaluate: compare ground truth with predictions
        # NerMetrics
        labels = ["micro", "macro"]
        metrics = ["precision", "recall", "f1"]
        metrics_seqeval = [f"{metric}_seqeval" for metric in metrics]
        levels = ["entity", "token"]
        evaluation: EVALUATION_DICT = {
            label: {
                level: {metric: None for metric in metrics + metrics_seqeval}
                for level in levels
            }
            for label in labels
        }

        ner_metrics_entity = NerMetrics(
            true_flat, pred_flat, level="entity", scheme="bio"
        )
        ner_metrics_entity.compute(metrics)
        # print("== ENTITY (nerblackbox) ==")
        for metric in metrics:
            _metric = metric if metric == "acc" else f"{metric}_micro"
            # print(f"> {metric}: {ner_metrics_entity.results_as_dict()[_metric]:.3f}")
        # print()

        for metric, label in product(metrics, labels):
            evaluation[label]["entity"][metric] = ner_metrics_entity.results_as_dict()[
                f"{metric}_{label}"
            ]

        # seqeval - just for testing - start
        from seqeval.metrics import precision_score, recall_score, f1_score

        scores = [precision_score, recall_score, f1_score]
        # print("== ENTITY (seqeval) ==")
        for metric_seqeval, score in zip(metrics_seqeval, scores):
            result = score(
                ground_truth,
                predictions,
            )
            # print(f"> {metric_seqeval}: {result:.3f}")
            evaluation["micro"]["entity"][f"{metric_seqeval}"] = result

        return evaluation

    def _analyze_tokenizer(self, tokenizer_config: Dict = {}) -> None:
        r"""
        retrieve information about tokenizer

        Created Attr:
            self.tokenizer_special: e.g. ["[CLS]", "[SEP]", "[PAD]"] for WordPiece, ['</s>', '<s>', '<pad>'] for SentencePiece
            self.tokenizer_add_prefix_space: True/False
            self.tokenizer_type: e.g. "WordPiece" or "SentencePiece"
        """
        # tokenizer_add_prefix_space
        add_prefix_space = tokenizer_config["add_prefix_space"] if "add_prefix_space" in tokenizer_config.keys() else None
        if add_prefix_space is not None:
            self.tokenizer_add_prefix_space = add_prefix_space
        else:
            self.tokenizer_add_prefix_space = False

        # tokenizer_type
        tokenizer_class = tokenizer_config["tokenizer_class"] if "tokenizer_class" in tokenizer_config.keys() else None
        self.tokenizer_type = None
        if tokenizer_class is not None:
            # explicitly set tokenizer type
            if tokenizer_class == "BertTokenizer":
                self.tokenizer_type = "WordPiece"
            elif tokenizer_class == "DistilBertTokenizer":
                self.tokenizer_type = "WordPiece"
            elif tokenizer_class == "ElectraTokenizer":
                self.tokenizer_type = "WordPiece"
            elif tokenizer_class == "RobertaTokenizer":
                self.tokenizer_type = "SentencePiece"
            elif tokenizer_class == "DebertaTokenizer":
                self.tokenizer_type = "SentencePiece"
            elif tokenizer_class == "AlbertTokenizer":
                self.tokenizer_type = "WordPiece"
                raise Exception(f"ERROR! tokenizer = {tokenizer_class} not supported.")
            elif tokenizer_class == "XLMRobertaTokenizer":
                self.tokenizer_type = "SentencePiece"
                raise Exception(f"ERROR! tokenizer = {tokenizer_class} not supported.")
            else:
                print(f"WARNING! tokenizer_class = {tokenizer_class} not directly supported."
                      f"Properties will be derived.")

        if self.tokenizer_type is None:
            # derive tokenizer type
            if self.tokenizer_add_prefix_space is True:
                self.tokenizer_type = "SentencePiece"
            else:
                self.tokenizer_type = "WordPiece"
            print(f"WARNING! tokenizer_type = {self.tokenizer_type} "
                  f"derived from add_prefix_space = {self.tokenizer_add_prefix_space}")

        # tokenizer_special
        def _extract_token(_token: str):
            if _token in tokenizer_config.keys():
                if isinstance(tokenizer_config[_token], str):
                    return tokenizer_config[_token]
                elif isinstance(tokenizer_config[_token], dict) and "content" in tokenizer_config[_token].keys():
                    return tokenizer_config[_token]["content"]
            return None

        bos_token = _extract_token("bos_token")
        eos_token = _extract_token("eos_token")
        sep_token = _extract_token("sep_token")
        pad_token = _extract_token("pad_token")
        cls_token = _extract_token("cls_token")

        self.tokenizer_special = [
            elem for elem in list({
                bos_token,
                eos_token,
                sep_token,
                pad_token,
                cls_token,
            })
            if elem is not None
        ]  # ["[CLS]", "[SEP]", "[PAD]"] or ['</s>', '<s>', '<pad>']
        if len(self.tokenizer_special) == 0:
            if self.tokenizer_type == "WordPiece":
                self.tokenizer_special = ["[CLS]", "[SEP]", "[PAD]"]
            else:
                self.tokenizer_special = ['</s>', '<s>', '<pad>']
            print(f"WARNING! tokenizer_special = {self.tokenizer_special} "
                  f"derived from tokenizer_type = {self.tokenizer_type}")


########################################################################################################################
########################################################################################################################
########################################################################################################################
def round_evaluation_dict(
    _evaluation_dict: EVALUATION_DICT, _rounded_decimals: int
) -> EVALUATION_DICT:
    r"""

    Args:
        _evaluation_dict: e.g.
        {
            "micro": {
                "entity": {
                    'precision': 0.9847222222222223,
                    'recall': 0.9916083916083916,
                    'f1': 0.9881533101045297,
                    'precision_seqeval': 0.9833564493758669,
                    'recall_seqeval': 0.9916083916083916,
                    'f1_seqeval': 0.9874651810584958,
                }
            }
        }
        _rounded_decimals: e.g. 3

    Returns:
        _evaluation_dict_rounded: e.g.
        {
            "micro": {
                "entity": {
                    'precision': 0.985,
                    'recall': 0.992,
                    'f1': 0.988,
                    'precision_seqeval': 0.983,
                    'recall_seqeval': 0.992,
                    'f1_seqeval': 0.987,
                }
            }
        }
    """
    return {
        k1: {
            k2: {
                k3: round(v3, _rounded_decimals) if isinstance(v3, float) else None
                for k3, v3 in v2.items()
            }
            for k2, v2 in v1.items()
        }
        for k1, v1 in _evaluation_dict.items()
    }


def derive_annotation_scheme(_id2label: Dict[int, str]) -> str:
    r"""
    Args:
        _id2label: e.g. {0: "O", 1: "B-PER", 2: "I-PER"}

    Returns:
        annotation_scheme: e.g. "bio"
    """
    if len(_id2label) == 0:
        raise Exception(
            f"ERROR! could not derive annotation scheme. id2label is empty."
        )

    label_prefixes = list(
        set([label.split("-")[0] for label in _id2label.values() if "-" in label])
    )
    if len(label_prefixes) == 0:
        return "plain"
    elif (
        "B" in label_prefixes
        and "I" in label_prefixes
        and "L" in label_prefixes
        and "U" in label_prefixes
    ):
        return "bilou"
    elif "B" in label_prefixes and "I" in label_prefixes:
        return "bio"
    elif "I" in label_prefixes:  # e.g. "Jean-Baptiste/camembert-ner"
        return "plain"
    else:
        raise Exception(
            f"ERROR! could not derive annotation scheme. found label_prefixes = {label_prefixes}"
        )


def turn_tensors_into_tag_probability_distributions(
    annotation_classes: List[str], outputs: torch.Tensor
) -> List[List[Dict[str, float]]]:
    """
    Args:
        annotation_classes: e.g. ["O", "B-PER", "I-PER"]
        outputs: [torch tensor]  of shape = [batch_size, seq_length, num_labels]

    Returns:
        predictions_proba: [list] of [list] of [prob dist], i.e. dict that maps tags to probabilities
    """
    probability_distributions = softmax(outputs, dim=2)
    predictions_proba = [
        [
            {
                annotation_classes[j]: float(
                    probability_distributions[x][i][j].detach().numpy()
                )
                for j in range(len(annotation_classes))
            }
            for i in range(outputs.shape[1])  # seq_length
        ]
        for x in range(outputs.shape[0])  # number_of_input_texts
    ]

    return predictions_proba


def merge_slices_for_single_document(_list: List[List[Any]]) -> List[Any]:
    """
    merges the slices for a single document

    input is
    - either List[List[str]], for tokens & predictions
    - or     List[List[Dict[str, float]]], for predictions with proba = True

    Args:
        _list: predictions for one or more slices that belong to same document,
               e.g. one slice:  [["[CLS]", "this", "is", "one", "slice", "[SEP]"]],
               e.g. two slices: [
                                    ["[CLS]", "this", "is", "one", "slice", "[SEP]"],
                                    ["[CLS]", "and", "a", "second", "one", "[SEP]"]
                                ],

    Returns:
        _list_flat: predictions for one document
               e.g. one slice:  ["[CLS]", "this", "is", "one", "slice", "[SEP]"],
               e.g. two slices: ["[CLS]", "this", "is", "one", "slice", "and", "a", "second", "one", "[SEP]"],
    """
    if len(_list) == 1:  # one slice
        return _list[0]
    else:  # more slices
        _list_flat = list()
        for i, sublist in enumerate(_list):
            if i == 0:
                _list_flat.extend(sublist[:-1])
            elif 0 < i < len(_list) - 1:
                _list_flat.extend(sublist[1:-1])
            else:
                _list_flat.extend(sublist[1:])
        return _list_flat


def merge_subtoken_to_token_predictions(
    tokens: List[str],
    predictions: List[Any],
    tokenizer_special: List[str],
    tokenizer_type: str,
) -> List[Tuple[str, Any]]:
    """
    Args:
        tokens: e.g. ["[CLS]", "arbetsförmedl", "##ingen", "finns", "i", "stockholm", "[SEP]", "[PAD]"]
        predictions:  e.g. ["[S]", "ORG", "ORG", "O", "O", "O", "[S]", "[S]"]
        tokenizer_special: e.g. ["[CLS]", "[SEP]", "[PAD]"] for WordPiece, ['</s>', '<s>', '<pad>'] for SentencePiece
        tokenizer_type: e.g. "WordPiece" or "SentencePiece"

    Returns:
        token_predictions: e.g. [("arbetsförmedlingen", "ORG"), ("finns", "O"), ("i", "O"), ("stockholm", "O")]
    """
    # predict tags on words between [CLS] and [SEP] or <s> and </s>
    token_predictions_list = list()
    for token, example_token_prediction in zip(tokens, predictions):
        if token not in tokenizer_special:
            if tokenizer_type == "WordPiece":
                if not token.startswith("##"):
                    token_predictions_list.append([token, example_token_prediction])
                else:
                    token_predictions_list[-1][0] += token.lstrip("##")
            elif tokenizer_type == "SentencePiece":
                if token.startswith("Ġ"):
                    token = token.strip("Ġ")
                    token_predictions_list.append([token, example_token_prediction])
                else:
                    token_predictions_list[-1][0] += token

    token_predictions = [(elem[0], elem[1]) for elem in token_predictions_list]

    return token_predictions


def restore_unknown_tokens(
    word_predictions: List[Tuple[str, Union[str, Dict[str, float]]]],
    input_text: str,
    verbose: bool = False,
) -> List[Dict[str, Union[str, Dict]]]:
    """
    - replace "[UNK]" tokens by the original token
    - enrich tokens with char_start & char_end

    Args:
        word_predictions: e.g. [('example', 'O'), ('of', 'O), ('author', 'PERSON'), ..]
        input_text: e.g. 'example of author'
        verbose: e.g. False

    Returns:
        word_predictions_restored: e.g. [
            {"char_start": "0", "char_end": "7", "token": "example", "tag": "O"},
            ..
        ]
    """
    word_predictions_restored = list()

    # 1. get margins of known tokens
    # word_predictions      = [('example', 'O'), ('of', 'O), ('author', 'PERSON'), ..]
    # input_text            = 'example of author ..'
    # => token_char_margins = [(0, 7), (8, 10), (11, 17), ..]
    token_char_margins: List[Tuple[Any, ...]] = list()
    char_start = 0
    unknown_counter = 0
    invalid_counter = 0
    for token, _ in word_predictions:
        if token == "[UNK]":
            token_char_margins.append((None, None))
            unknown_counter += 1
        else:
            # move char_start to where punctuation or whitespace is found
            while unknown_counter > 0:
                try:
                    if token in string.punctuation or len(token) != 1:
                        char_start = input_text.index(token, char_start)
                    else:  # i.e. len(token) == 1
                        char_start = input_text.index(f" {token}", char_start)
                except ValueError:  # .index() did not find anything
                    if DEBUG:
                        print(
                            f"! token = {token} not found in example[{char_start}:] (unknown)"
                        )
                unknown_counter -= 1

            # find token_char_margins for token
            try:
                char_start_before = copy.deepcopy(char_start)
                # dirty method to find start of 2nd whitespace character after char_start
                _temp = input_text[char_start_before:].replace(' ', '-', 1).find(' ')
                whitespace_start_2nd = _temp + char_start_before if _temp > -1 else -1
                char_start = input_text.index(token, char_start_before)
                whitespaces_before_char_start = len(input_text[:char_start]) - len(input_text[:char_start].rstrip())
                whitespaces_after_char_start = len(input_text[char_start:]) - len(input_text[char_start:].lstrip())
                whitespaces_around_char_start = whitespaces_before_char_start + whitespaces_after_char_start
                if DEBUG:
                    print(f"! token = {token.ljust(20)}, "
                          f"char_start_before = {char_start_before}, "
                          f"char_start = {char_start}, "
                          f"whitespace_start_2nd = {whitespace_start_2nd}, "
                          f"invalid_counter = {invalid_counter}",
                          f"whitespaces_before_char_start = {whitespaces_before_char_start}, "
                          f"whitespaces_after_char_start = {whitespaces_after_char_start}, "
                          f"len(token) = {len(token)}, ",
                    )
                valid = char_start <= char_start_before + invalid_counter + whitespaces_around_char_start and \
                    (whitespace_start_2nd == -1 or whitespace_start_2nd > char_start)
                if valid:
                    token_char_margins.append((char_start, char_start + len(token)))
                    invalid_counter = 0
                else:
                    invalid_counter += 1
                    if DEBUG:
                        print(f"! token = {token} not found in example[{char_start}:]")
                    char_start = char_start_before
                    token_char_margins.append((None, None))
            except ValueError:
                invalid_counter += 1
                if DEBUG:
                    print(f"! token = {token} not found in example[{char_start}:]")
                token_char_margins.append((None, None))
            char_start += len(token)
            unknown_counter = 0

    # 2. restore unknown tokens
    # word_predictions   = [('example', 'O'), ('of', 'O), ('author', 'PERSON'), ..]
    # input_text         = 'example of author ..'
    # token_char_margins = [(0, 7), (8, 10), (11, 17), ..]
    # => word_predictions_restored = [{"char_start": "0", "char_end": "7", "token": "example", "tag": "O"}, ..]
    unresolved_margins = list()
    for i, (token, tag) in enumerate(word_predictions):
        if (
            token_char_margins[i][0] is not None
            and token_char_margins[i][1] is not None
        ):
            word_predictions_restored.append(
                {
                    "char_start": str(token_char_margins[i][0]),
                    "char_end": str(token_char_margins[i][1]),
                    "token": token,
                    "tag": tag,
                }
            )
        else:
            for j in range(2):
                assert (
                    token_char_margins[i][j] is None
                ), f"ERROR! token_char_margin[{i}][{j}] is not None for token = {token}"

            char_start_margin, char_end_margin = None, None
            # k_prev = steps until first known token to the left  is reached
            # k_next = steps until first known token to the right is reached
            # char_start_margin = start of unknown span
            # char_end_margin   = end   of unknown span
            k_prev, k_next = None, None
            for k in range(1, 10):
                k_prev = k
                if i - k < 0:
                    char_start_margin = 0
                    break
                elif token_char_margins[i - k][1] is not None:
                    char_start_margin = token_char_margins[i - k][1]
                    break
            for k in range(1, 10):
                k_next = k
                if i + k >= len(token_char_margins):
                    char_end_margin = len(input_text)
                    break
                elif token_char_margins[i + k][0] is not None:
                    char_end_margin = token_char_margins[i + k][0]
                    break
            assert (
                char_start_margin is not None
            ), f"ERROR! could not find char_start_margin"
            assert char_end_margin is not None, (
                f"ERROR! could not find char_end_margin. token = {token}\n"
                f"char_start_margin = {char_start_margin}\n"
                f"char_end_margin = {char_end_margin}\n"
                f"input_text = {input_text}\n"
                f"token_char_margins = {token_char_margins}\n"
                f"word_predictions = {word_predictions}\n"
                f"word_predictions_restored = {word_predictions_restored}"
            )
            assert (
                k_prev is not None and k_next is not None
            ), f"ERROR! could not find k_prev or k_next"

            new_token = input_text[char_start_margin:char_end_margin].strip()
            if k_prev != 1 or k_next != 1:
                new_token_parts = new_token.split()
                assertion_1 = len(new_token_parts) == -1 + np.absolute(k_prev + k_next)
                if not assertion_1:
                    if DEBUG:
                        print(
                            f"--------- WARNING START --------\n",
                            f"ERROR! new_token = {new_token} consists of {len(new_token_parts)} parts, "
                            f"expected {1 + np.absolute(k_prev - k_next)} (k_prev = {k_prev}, k_next = {k_next})\n"
                            f"char_start_margin = {char_start_margin}\n"
                            f"char_end_margin = {char_end_margin}\n"
                            f"input_text = {input_text}\n"
                            f"token_char_margins = {token_char_margins}\n"
                            f"word_predictions = {word_predictions}\n"
                            f"word_predictions_restored = {word_predictions_restored}\n"
                            f"new_token = {new_token}\n"
                            f"new_token_parts = {new_token_parts}\n"
                            f"--------- WARNING END --------\n",
                        )
                    new_token = ""
                else:
                    new_token = new_token_parts[k_prev - 1]

            if len(new_token):
                char_start = input_text.index(new_token, char_start_margin)
                char_end = char_start + len(new_token)
                if DEBUG:
                    print(
                        f"! restored unknown token = {new_token} between "
                        f"char_start = {char_start}, "
                        f"char_end = {char_end}"
                    )
                word_predictions_restored.append(
                    {
                        "char_start": str(char_start),
                        "char_end": str(char_end),
                        "token": new_token,
                        "tag": tag,
                    }
                )
                token_char_margins[i] = (char_start, char_end)  # for next iteration
            else:
                if DEBUG:
                    print(
                        f"!!! could not restore unknown token between "
                        f"char_start_margin = {char_start_margin}, "
                        f"char_end_margin = {char_end_margin}"
                    )
                unresolved_margins.append((char_start_margin, char_end_margin))

    # 3. resolve unresolved margins
    unresolved_margins = list(set(unresolved_margins))
    if len(unresolved_margins):
        for (char_start_margin, char_end_margin) in unresolved_margins:
            words_to_restore = input_text[char_start_margin:char_end_margin].split()
            char_start_temp = char_start_margin
            for word_to_restore in words_to_restore:
                char_start = char_start_temp + input_text[char_start_temp:char_end_margin].index(word_to_restore)
                char_end = char_start + len(word_to_restore)
                char_start_temp = char_end
                word_predictions_restored.append(
                    {
                        "char_start": str(char_start),
                        "char_end": str(char_end),
                        "token": word_to_restore,
                        "tag": "O",
                    }
                )
                print(f"WARNING! couldn't restore tokens. restored word w/ tag = O: {word_predictions_restored[-1]}")
        word_predictions_restored = sorted(word_predictions_restored, key=lambda d: int(d['char_start']))

    return word_predictions_restored


def assert_typing(
    input_text_word_predictions: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    this is only to ensure correct typing, it does not actually change anything

    Args:
        input_text_word_predictions: e.g. [
            {"char_start": 0, "char_end": 7, "token": "example", "tag": "O"},
            ..
        ]

    Returns:
        input_text_word_predictions_str: e.g. [
            {"char_start": "0", "char_end": "7", "token": "example", "tag": "O"},
            ..
        ]
    """
    return [
        {k: str(v) for k, v in input_text_word_prediction.items()}
        for input_text_word_prediction in input_text_word_predictions
    ]
