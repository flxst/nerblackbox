from nerblackbox.modules.ner_training.data_preprocessing.tools.bert_dataset import (
    BertDataset,
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
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.modules.utils.util_functions import get_dataset_path
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler
import pandas as pd
from pkg_resources import resource_filename

from typing import List, Dict, Tuple, Optional, Any

InputExamples = List[InputExample]


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

    def get_input_examples_train(
        self, prune_ratio: Dict[str, float], dataset_name: Optional[str] = None
    ) -> Tuple[Dict[str, InputExamples], List[str]]:
        """
        - get input examples for TRAIN from csv files

        Args:
            dataset_name:     [str], e.g. 'suc'
            prune_ratio:      [dict], e.g. {'train': 1.0, 'val': 1.0, 'test': 1.0} -- pruning ratio for data

        Returns:
            input_examples:   [dict] w/ keys = 'train', 'val', 'test' & values = [list] of [InputExample]
            tag_list_ordered: [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
        """
        if dataset_name is None:
            dataset_path = resource_filename("nerblackbox", "tests/test_data")
        else:
            dataset_path = get_dataset_path(dataset_name)

        # csv_reader
        csv_reader = CsvReader(
            dataset_path,
            self.tokenizer,
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

        tag_list = self._ensure_completeness_in_case_of_bio_tags(csv_reader.tag_list)
        tag_list_ordered = order_tag_list(tag_list)

        return input_examples, tag_list_ordered

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
        # print('TEMP: input_examples:', input_examples)

        return input_examples

    def to_dataloader(
        self,
        input_examples: Dict[str, InputExamples],
        tag_list: List[str],
        batch_size: int,
    ) -> Dict[str, DataLoader]:
        """
        - turn input_examples into dataloader

        Args:
            input_examples: [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                      values = [list] of [InputExample]
            tag_list:       [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
            batch_size:     [int]

        Returns:
            _dataloader:    [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                      values = [torch Dataloader]
        """
        # input_example_to_tensors
        input_examples_to_tensors = InputExamplesToTensors(
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            tag_tuple=tuple(tag_list),
            default_logger=self.default_logger,
        )

        _dataloader = dict()
        for phase in input_examples.keys():
            encodings = input_examples_to_tensors(
                input_examples[phase], predict=phase == "predict"
            )

            # dataloader
            data = BertDataset(
                encodings=encodings
            )  # data[j] = 4 torch tensors corresponding to EncodingKeys
            if self.default_logger:
                self.default_logger.log_info(
                    f"[after pre-preprocessing] {phase.ljust(5)} data: {len(data)} examples"
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
            num_examples_new = int(ratio * float(num_examples_old))
            info = f"> {phase.ljust(5)} data: use {num_examples_new} of {num_examples_old} examples"
            if self.default_logger:
                self.default_logger.log_info(info)
            return list_of_examples[:num_examples_new]

    @staticmethod
    def _ensure_completeness_in_case_of_bio_tags(tag_list: List[str]) -> List[str]:
        """
        make sure that there is an "I-*" tag for every "B-*" tag in case of BIO-tags
        ----------------------------------------------------------------------------
        :param tag_list:             e.g. ["B-person", "B-time", "I-person"]
        :return: completed_tag_list: e.g. ["B-person", "B-time", "I-person", "I-time"]
        """
        b_tags = [tag for tag in tag_list if tag.startswith("B")]
        for b_tag in b_tags:
            i_tag = b_tag.replace("B-", "I-")
            if i_tag not in tag_list:
                tag_list.append(i_tag)
        return tag_list


def order_tag_list(tag_list: List[str]) -> List[str]:
    return ["O"] + sorted([elem for elem in tag_list if elem != "O"])


def convert_tag_list_bio2plain(tag_list_bio: List[str]) -> List[str]:
    tag_list_bio_without_o = [elem for elem in tag_list_bio if elem != "O"]
    return ["O"] + sorted(
        pd.Series(tag_list_bio_without_o)
        .map(lambda x: x.split("-")[-1])
        .drop_duplicates()
        .tolist()
    )
