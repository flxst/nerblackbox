import json
import pandas as pd

from nerblackbox.modules.ner_training.data_preprocessing.tools.encodings_dataset import (
    EncodingsDataset,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.csv_reader import (
    CsvReader,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_examples_to_tensors import (
    InputExamplesToTensors,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    InputExamples,
)
from nerblackbox.modules.ner_training.annotation_tags.annotation import Annotation
from nerblackbox.modules.ner_training.logging.default_logger import DefaultLogger
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.modules.utils.util_functions import get_dataset_path
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler
from pkg_resources import resource_filename
from copy import deepcopy
from os.path import join, isfile
from typing import List, Dict, Tuple, Optional, Any, Union

PHASES = ["train", "val", "test"]
SENTENCES_ROWS_UNPRETOKENIZED = List[Dict[str, Any]]


class DataPreprocessor:
    def __init__(
        self,
        tokenizer,
        do_lower_case: bool,
        default_logger: Union[DefaultLogger, PseudoDefaultLogger],
        max_seq_length: int = 64,
    ):
        """
        :param tokenizer:      [transformers Tokenizer]
        :param do_lower_case:  [bool] if True, make text data lowercase
        :param default_logger: [DefaultLogger]
        :param max_seq_length: [int], e.g. 64
        """
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case
        self.default_logger = default_logger
        self.max_seq_length = max_seq_length
        self.pretokenized: Optional[
            bool
        ] = None  # whether dataset is pretokenized (csv) or not (jsonl)

    def get_input_examples_train(
        self,
        prune_ratio: Dict[str, float],
        dataset_name: Optional[str] = None,
        train_on_val: Optional[bool] = None,
        train_on_test: Optional[bool] = None,
    ) -> Tuple[Dict[str, InputExamples], Annotation]:
        """
        - get input examples for TRAIN from csv files

        Args:
            prune_ratio:      [dict], e.g. {'train': 1.0, 'val': 1.0, 'test': 1.0} -- pruning ratio for data
            dataset_name:     [str], e.g. 'suc'
            train_on_val:     [bool] if True, train additionally on validation set
            train_on_test:    [bool] if True, train additionally on test set

        Returns:
            input_examples:          [dict] w/ keys = 'train', 'val', 'test' & values = [list] of [InputExample]
            annotation:              [Annotation] instance
        """
        if dataset_name is None:
            dataset_path = resource_filename("nerblackbox", "tests/test_data")
        else:
            dataset_path = get_dataset_path(dataset_name)

        self.pretokenized = self._check_if_data_is_pretokenized(dataset_path)
        if not self.pretokenized:
            self._pretokenize(
                dataset_path
            )  # "{phase}.jsonl" -> "pretokenized_{phase}.csv"

        csv_reader = CsvReader(
            dataset_path,
            self.tokenizer,
            pretokenized=self.pretokenized,
            do_lower_case=self.do_lower_case,  # can be True (applies .lower()) !!
            default_logger=self.default_logger,
        )

        input_examples = dict()
        for phase in ["train", "val", "test"]:
            # train data
            input_examples_all = csv_reader.get_input_examples(phase)
            input_examples[phase] = self._prune_examples(
                input_examples_all, phase, ratio=prune_ratio[phase]
            )

        # note: w/o deepcopy, potential annotation scheme conversion fails
        # (as it will be applied twice in InputExamplesUtils.convert_annotation_scheme())
        if train_on_val:
            input_examples["train"] += deepcopy(input_examples["val"])
        if train_on_test:
            input_examples["train"] += deepcopy(input_examples["test"])

        annotation = Annotation(csv_reader.annotation_classes)

        return input_examples, annotation

    def get_input_examples_predict(
        self,
        examples: List[str],
        is_pretokenized: bool,
    ) -> Tuple[
        Dict[str, InputExamples], List[str], Optional[List[List[Tuple[int, int]]]]
    ]:
        """
        - get input examples for PREDICT from input argument examples

        Args:
            examples:       [list] of [str]
            is_pretokenized: True if examples are pretokenized

        Returns:
            input_examples:         [dict] w/ key = 'predict' & value = [list] of [InputExample]
            examples_pretokenized:  [list] of [str]
            pretokenization_offsets: [list] of [list] of [tuples]
        """
        # create input_examples
        if self.do_lower_case:
            examples = [example.lower() for example in examples]

        if is_pretokenized:
            _data_pretokenized = [
                {
                    "text": example,
                    "tags": " ".join(
                        ["O" for _ in range(len(example.split()))]
                    ),  # pseudo tags
                }
                for example in examples
            ]
            pretokenization_offsets = None
        else:
            _data = [
                {
                    "text": example,
                    "tags": [],
                }
                for example in examples
            ]
            _data_pretokenized, pretokenization_offsets = self._pretokenize_data(_data)

        examples_pretokenized = [elem["text"] for elem in _data_pretokenized]

        input_examples = {
            "predict": [
                InputExample(
                    guid="",
                    text=elem["text"],
                    tags=elem["tags"],
                )
                for elem in _data_pretokenized
            ]
        }

        return input_examples, examples_pretokenized, pretokenization_offsets

    def to_dataloader(
        self,
        input_examples: Dict[str, InputExamples],
        annotation_classes: List[str],
        batch_size: int,
    ) -> Tuple[Dict[str, DataLoader], Dict[str, List[int]]]:
        """
        - turn input_examples into dataloader

        Args:
            input_examples:     [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                          values = [list] of [InputExample]
            annotation_classes: [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
            batch_size:         [int]

        Returns:
            _dataloader:    [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                      values = [torch Dataloader]
            _offsets:        [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                       values = [List] of [int]
                                       which gives information on how the input_examples where slices,
                                       e.g. offsets = [0, 2, 3]
                                       means that the first two slices belong to the first input example (0->2),
                                       while the third slice belongs to the second input example (2->3).
        """
        # input_example_to_tensors
        input_examples_to_tensors = InputExamplesToTensors(
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            annotation_classes_tuple=tuple(annotation_classes),
            default_logger=self.default_logger,
        )

        _dataloader = dict()
        _offsets = dict()
        for phase in input_examples.keys():
            self.default_logger.log_info(
                f"[before preprocessing] {phase.ljust(5)} data: {len(input_examples[phase])} examples"
            )
            encodings, _offsets_phase = input_examples_to_tensors(
                input_examples[phase], predict=phase == "predict"
            )
            _offsets[phase] = _offsets_phase

            # dataloader
            data = EncodingsDataset(
                encodings=encodings
            )  # data[j] = 4 torch tensors corresponding to EncodingKeys
            if self.default_logger:
                self.default_logger.log_info(
                    f"[after  preprocessing] {phase.ljust(5)} data: {len(data)} examples"
                )

            assert phase in [
                "train",
                "val",
                "test",
                "predict",
            ], f"ERROR! phase = {phase} unknown."
            sampler: Optional[Sampler]
            if phase == "train":
                sampler = RandomSampler(data)
            elif phase in ["val", "test"]:
                sampler = SequentialSampler(data)
            else:  # if phase == "predict":
                sampler = None

            _dataloader[phase] = DataLoader(
                data,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=0,  # num_workers=1, pin_memory=True
                # see https://pytorch-lightning.readthedocs.io/en/stable/benchmarking/performance.html
            )

        return _dataloader, _offsets

    ####################################################################################################################
    # HELPER
    ####################################################################################################################
    @staticmethod
    def _check_if_data_is_pretokenized(dataset_path: str) -> bool:
        """
        check if dataset_name is pretokenized (csv) or not (jsonl)

        Args:
            dataset_path: path to dataset directory

        Returns:
            pretokenized: whether dataset is pretokenized (csv) or not (jsonl)
        """
        dataset_path_csv = join(dataset_path, f"train.csv")
        dataset_path_jsonl = join(dataset_path, f"train.jsonl")
        if isfile(dataset_path_csv) and not isfile(dataset_path_jsonl):
            return True
        elif not isfile(dataset_path_csv) and isfile(dataset_path_jsonl):
            return False
        else:
            raise Exception(
                f"ERROR! Did not find train.csv OR train.jsonl in {dataset_path}:"
                f"(csv: {isfile(dataset_path_csv)}, jsonl: {isfile(dataset_path_jsonl)})"
            )

    @staticmethod
    def _tokens2words(_tokens: List[str]) -> List[str]:
        """
        merges tokens to words

        Args:
            _tokens: e.g. ["this", "example", "contains", "hu", "##gging", "face"]

        Returns:
            _words: e.g. ["this", "example", "contains", "hugging", "face"]
        """
        i = 1
        while i < len(_tokens):
            if _tokens[i].startswith("##"):
                _tokens[i - 1] += _tokens.pop(i).strip("##")  # join with previous
            else:
                i += 1
        return _tokens

    @staticmethod
    def _resolve_overlapping_tags(_tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """

        Args:
            _tags: e.g.
            [
             {"token": "Bajo peso", "tag": "Concept", "char_start": 4651, "char_end": 4660},
             {"token": "peso", "tag": "Concept", "char_start": 4656, "char_end": 4660},
             {"token": "más", "tag": "Predicate", "char_start": 4681, "char_end": 4684},
            ]

        Returns:
            _resolved_tags: e.g.
            [
             {"token": "Bajo peso", "tag": "Concept", "char_start": 4651, "char_end": 4660},
             {"token": "más", "tag": "Predicate", "char_start": 4681, "char_end": 4684},
            ]
        """
        _resolved_tags = list()
        for i in range(len(_tags)):
            if i == 0 or _tags[i - 1]["char_end"] <= _tags[i]["char_start"]:
                _resolved_tags.append(_tags[i])
        return _resolved_tags

    def _pretokenize_data(
        self, _data: SENTENCES_ROWS_UNPRETOKENIZED
    ) -> Tuple[List[Dict[str, str]], List[List[Tuple[int, int]]]]:
        """
        pretokenize data (text and text simultaneously)

        Args:
            _data: e.g. [
                {
                    'text': 'arbetsförmedlingen ai-center finns i stockholm.',
                    'tags': [
                        {"token": "arbetsförmedlingen ai-center", "tag": "ORG", "char_start": 0, "char_end": 28},
                        {"token": "stockholm", "tag": "LOC", "char_start": 37, "char_end": 46},
                    ]
                },
                ..
            ]

        Returns:
            _data_pretokenized: e.g. [
                {
                    'text': 'arbetsförmedlingen ai - center finns i stockholm .',
                    'tags': 'B-ORG I-ORG I-ORG I-ORG O O B-LOC O'
                },
                ..
            ]
            _pretokenization_offsets: e.g. [
                [(0,17), (18,20), (20,21), (21,26)]
            ]
        """
        _data_pretokenized = list()
        _pretokenization_offsets = list()
        for n in range(len(_data)):
            word_tuples = (
                self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                    _data[n]["text"]
                )
            )
            tags = ["O"] * len(word_tuples)
            entity_dicts = self._resolve_overlapping_tags(_data[n]["tags"])
            for entity_dict in entity_dicts:
                entity_char_start = entity_dict["char_start"]
                entity_char_end = entity_dict["char_end"]
                entity_tag = entity_dict["tag"]
                for word_index, word_tuple in enumerate(word_tuples):
                    word_char_start = word_tuple[1][0]
                    word_char_end = word_tuple[1][1]
                    if (
                        word_char_start == entity_char_start
                        and word_char_end <= entity_char_end
                    ):
                        tags[word_index] = f"B-{entity_tag}"
                    elif (
                        word_char_start >= entity_char_start
                        and word_char_end <= entity_char_end
                    ):
                        tags[word_index] = f"I-{entity_tag}"

            words = [word_tuple[0] for word_tuple in word_tuples]
            pretokenization_offsets = [word_tuple[1] for word_tuple in word_tuples]
            _pretokenization_offsets.append(pretokenization_offsets)

            _data_pretokenized.append(
                {
                    "tags": " ".join(tags),
                    "text": " ".join(words),
                }
            )

        return _data_pretokenized, _pretokenization_offsets

    def _pretokenize(self, dataset_path: str) -> None:  # pragma: no cover
        """
        1. read jsonl files "{phase}.jsonl"
        2. pretokenize data
        3. write csv files "pretokenized_{phase}.csv"

        Args:
            dataset_path: path to dataset directory

        Returns: -
        """
        for phase in PHASES:
            # 1. read json files
            dataset_path_jsonl = join(dataset_path, f"{phase}.jsonl")
            with open(dataset_path_jsonl, "r") as f:
                jlines = f.readlines()
                data = [json.loads(jline) for jline in jlines]

            # 2. pretokenize data
            data_pretokenized, _ = self._pretokenize_data(data)

            # 3. write csv files "pretokenized_{phase}.csv"
            df = pd.DataFrame(data_pretokenized)
            file_path = join(dataset_path, f"pretokenized_{phase}.csv")
            df.to_csv(file_path, sep="\t", header=False, index=False)

    def _prune_examples(self, list_of_examples: List[Any], phase: str, ratio: float):
        """
        prunes list_of_examples by taking only the first examples
        ---------------------------------------------------------
        :param list_of_examples: [list], e.g. of [InputExample]
        :param phase:            [str], 'train' or 'valid'
        :param (Optional) ratio: [float], e.g. 0.5
        :return: [list], e.g. of [InputExample]
        """
        if ratio == 1.0:
            return list_of_examples
        else:
            num_examples_old = len(list_of_examples)
            num_examples_new = max(1, int(ratio * float(num_examples_old)))
            info = f"> {phase.ljust(5)} data: use {num_examples_new} of {num_examples_old} examples"
            if self.default_logger:
                self.default_logger.log_info(info)
            return list_of_examples[:num_examples_new]
