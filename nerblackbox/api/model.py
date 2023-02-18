import json
import string
from os.path import join, isdir, isfile
from typing import Tuple, Any, Optional
import numpy as np

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List, Union, Dict
from torch.nn.functional import softmax
from huggingface_hub import hf_hub_download
from itertools import product

from nerblackbox.modules.ner_training.annotation_tags.token_tags import TokenTags
from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.api.store import Store
from nerblackbox.api.dataset import Dataset
from nerblackbox.modules.ner_training.metrics.ner_metrics import NerMetrics
from nerblackbox.modules.experiment_results import ExperimentResults
from nerblackbox.modules.ner_training.ner_model_train2model import (
    NerModelTrain2Model,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.csv_reader import (
    CsvReader,
)
from nerblackbox.modules.ner_training.annotation_tags.tags import Tags

PREDICTIONS = List[List[Dict[str, Any]]]
EVALUATION_DICT = Dict[str, Dict[str, Dict[str, Optional[float]]]]

VERBOSE = False


class Model:
    r"""
    model that predicts tags for given input text
    """

    @classmethod
    def from_experiment(cls, experiment_name: str) -> Optional["Model"]:
        r"""load best model from experiment.

        Args:
            experiment_name: name of the experiment, e.g. "exp0"

        Returns:
            model: best model from experiment
        """
        experiment_exists, experiment_results = Store.get_experiment_results_single(
            experiment_name
        )
        assert (
            experiment_exists
        ), f"ERROR! experiment = {experiment_name} does not exist."

        assert isinstance(
            experiment_results, ExperimentResults
        ), f"ERROR! experiment_results expected to be an instance of ExperimentResults."

        if experiment_results.best_single_run is None:
            print(
                f"> ATTENTION! could not find results for experiment = {experiment_name}"
            )
            return None
        elif (
            "checkpoint" not in experiment_results.best_single_run.keys()
            or experiment_results.best_single_run["checkpoint"] is None
        ):
            print(
                f"> ATTENTION! there is no checkpoint for experiment = {experiment_name}."
            )
            return None
        else:
            checkpoint_path_train = experiment_results.best_single_run["checkpoint"]
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
            model: best model from experiment
        """
        if not cls.checkpoint_exists(checkpoint_directory):
            print(
                f"> ATTENTION! could not find checkpoint directory at {checkpoint_directory}"
            )
            return None
        else:
            return Model(checkpoint_directory)

    @classmethod
    def from_huggingface(cls, repo_id: str) -> Optional["Model"]:
        r"""

        Args:
            repo_id: id of the huggingface hub repo id, e.g. 'KB/bert-base-swedish-cased-ner'

        Returns:
            model: model
        """
        # download files and get local file paths in cache directory
        filenames = [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]
        local_file_paths = []
        for filename in filenames:
            local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            local_file_paths.append(local_file_path)
        assert len(local_file_paths) == len(
            filenames
        ), f"ERROR! #local_file_paths = {len(local_file_paths)} does not correspond to #files = {len(filenames)}."

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
        return Model(cache_directory)

    @classmethod
    def checkpoint_exists(cls, checkpoint_directory: str) -> bool:
        return isdir(checkpoint_directory)

    def __init__(
        self,
        checkpoint_directory: str,
        batch_size: int = 16,
        max_seq_length: Optional[int] = None,
    ):
        r"""
        Args:
            checkpoint_directory: path to the checkpoint directory
            batch_size: batch size used by dataloader
            max_seq_length: maximum sequence length (Optional). Loaded from checkpoint if not specified.
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

        id2label = {int(_id): label for _id, label in config["id2label"].items()}
        label2id = {label: int(_id) for _id, label in config["id2label"].items()}
        self.annotation_classes = list(config["id2label"].values())

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_directory,
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
            do_lower_case=False,
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
            merge_slices_for_single_document(tokens[offsets[i] : offsets[i + 1]])
            for i in range(len(offsets) - 1)
        ]  # List[List[str]] with len = number_of_input_texts
        predictions = [
            merge_slices_for_single_document(predictions[offsets[i] : offsets[i + 1]])
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
            input_text_word_predictions: ???
        """
        ######################################
        # 1 input_text, merged chunks, tokens -> words
        ######################################
        _token_predictions: List[
            Tuple[str, Union[str, Dict[str, float]]]
        ] = merge_subtoken_to_token_predictions(tokens, predictions)

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
        dataset = Dataset(dataset_name)
        dataset.set_up()
        dir_path = f"{Store.get_path()}/datasets/{dataset_name}"
        return self._evaluate_on_csv(
            dir_path,
            phase=phase,
            class_mapping=class_mapping,
            number=number,
            derived_from_jsonl=False,
        )

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
            self.tokenizer,
            pretokenized=derived_from_jsonl is False,
            do_lower_case=self.tokenizer.do_lower_case,
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
) -> List[Tuple[str, Any]]:
    """
    Args:
        tokens: e.g. ["[CLS]", "arbetsförmedl", "##ingen", "finns", "i", "stockholm", "[SEP]", "[PAD]"]
        predictions:  e.g. ["[S]", "ORG", "ORG", "O", "O", "O", "[S]", "[S]"]

    Returns:
        token_predictions: e.g. [("arbetsförmedlingen", "ORG"), ("finns", "O"), ("i", "O"), ("stockholm", "O")]
    """
    # predict tags on words between [CLS] and [SEP]
    token_predictions_list = list()
    for token, example_token_prediction in zip(tokens, predictions):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            if not token.startswith("##"):
                token_predictions_list.append([token, example_token_prediction])
            else:
                token_predictions_list[-1][0] += token.strip("##")

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
    token_char_margins: List[Tuple[Any, ...]] = list()
    char_start = 0
    unknown_counter = 0
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
                    if verbose:
                        print(
                            f"! token = {token} not found in example[{char_start}:] (unknown)"
                        )
                unknown_counter -= 1

            # find token_char_margins for token
            try:
                char_start = input_text.index(token, char_start)
                token_char_margins.append((char_start, char_start + len(token)))
            except ValueError:
                if verbose:
                    print(f"! token = {token} not found in example[{char_start}:]")
                token_char_margins.append((None, None))
            char_start += len(token)
            unknown_counter = 0

    # 2. restore unknown tokens
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
                f"word_predictions = {word_predictions}\n"
                f"input_text = {input_text}\n"
                f"token_char_margins = {token_char_margins}\n"
                f"word_predictions_restored = {word_predictions_restored}"
            )
            assert (
                k_prev is not None and k_next is not None
            ), f"ERROR! could not find k_prev or k_next"

            new_token = input_text[char_start_margin:char_end_margin].strip()
            if k_prev != 1 or k_next != 1:
                new_token_parts = new_token.split()
                assert len(new_token_parts) == 1 + np.absolute(k_prev - k_next), (
                    f"ERROR! new_token = {new_token} consists of {len(new_token_parts)} parts, "
                    f"expected {1+np.absolute(k_prev-k_next)} (k_prev = {k_prev}, k_next = {k_next})"
                )
                new_token = new_token_parts[k_prev - 1]

            if len(new_token):
                char_start = input_text.index(new_token, char_start_margin)
                char_end = char_start + len(new_token)
                if verbose:
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
            else:
                if verbose:
                    print(
                        f"! dropped unknown empty token between "
                        f"char_start = {char_start_margin}, "
                        f"char_end = {char_end_margin}"
                    )

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
