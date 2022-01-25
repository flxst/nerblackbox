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
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.modules.utils.util_functions import get_dataset_path
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler
from pkg_resources import resource_filename
from copy import deepcopy
from os.path import join, isfile
from typing import List, Dict, Tuple, Optional, Any

PHASES = ["train", "val", "test"]
SENTENCES_ROWS_UNPRETOKENIZED = List[Dict[str, Any]]


class DataPreprocessor:
    def __init__(
        self,
        tokenizer,
        do_lower_case: bool,
        default_logger: PseudoDefaultLogger,
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
        self, examples: List[str]
    ) -> Dict[str, InputExamples]:
        """
        - get input examples for PREDICT from input argument examples

        Args:
            examples:       [list] of [str]

        Returns:
            input_examples: [dict] w/ key = 'predict' & value = [list] of [InputExample]
        """
        # create input_examples
        if self.do_lower_case:
            examples = [example.lower() for example in examples]

        input_examples = {
            "predict": [
                InputExample(
                    guid="",
                    text=example,
                    tags=" ".join(
                        ["O" for _ in range(len(example.split()))]
                    ),  # pseudo tags
                )
                for example in examples
            ]
        }

        return input_examples

    def to_dataloader(
        self,
        input_examples: Dict[str, InputExamples],
        annotation_classes: List[str],
        batch_size: int,
    ) -> Dict[str, DataLoader]:
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
        """
        # input_example_to_tensors
        input_examples_to_tensors = InputExamplesToTensors(
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            annotation_classes_tuple=tuple(annotation_classes),
            default_logger=self.default_logger,
        )

        _dataloader = dict()
        for phase in input_examples.keys():
            self.default_logger.log_info(
                f"[before preprocessing] {phase.ljust(5)} data: {len(input_examples[phase])} examples"
            )
            encodings = input_examples_to_tensors(
                input_examples[phase], predict=phase == "predict"
            )

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

        return _dataloader

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
                f"(csv: {isfile(dataset_path_csv)}, jsonl: {isfile(dataset_path_jsonl)}"
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

    def _pretokenize_data(
        self, _data: SENTENCES_ROWS_UNPRETOKENIZED
    ) -> List[Dict[str, str]]:
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
        """
        _data_pretokenized = list()
        for n in range(len(_data)):
            words = self._tokens2words(self.tokenizer.tokenize(_data[n]["text"]))
            index = 0
            tags = ["O"] * len(words)
            for entity_dict in _data[n]["tags"]:
                entity_text = entity_dict["token"]
                entity_words = self._tokens2words(self.tokenizer.tokenize(entity_text))
                entity_words_indices = [
                    index + words[index:].index(entity_word)
                    for entity_word in entity_words
                ]
                for i, entity_word_index in enumerate(entity_words_indices):
                    tags[entity_word_index] = (
                        f"B-{entity_dict['tag']}"
                        if i == 0
                        else f"I-{entity_dict['tag']}"
                    )
                index = max(entity_words_indices)

            _data_pretokenized.append(
                {
                    "tags": " ".join(tags),
                    "text": " ".join(words),
                }
            )

        return _data_pretokenized

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
            data_pretokenized = self._pretokenize_data(data)

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
