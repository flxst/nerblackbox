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

from typing import List, Dict, Tuple, Optional, Any


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
        self, prune_ratio: Dict[str, float], dataset_name: Optional[str] = None, train_on_val: Optional[bool] = None,
    ) -> Tuple[Dict[str, InputExamples], Annotation]:
        """
        - get input examples for TRAIN from csv files

        Args:
            prune_ratio:      [dict], e.g. {'train': 1.0, 'val': 1.0, 'test': 1.0} -- pruning ratio for data
            dataset_name:     [str], e.g. 'suc'
            train_on_val:     [bool] if True, train on training and validation set

        Returns:
            input_examples:          [dict] w/ keys = 'train', 'val', 'test' & values = [list] of [InputExample]
            annotation:              [Annotation] instance
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

        if train_on_val:
            input_examples["train"] += input_examples["val"]

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
