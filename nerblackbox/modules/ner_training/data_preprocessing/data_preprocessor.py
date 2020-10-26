from nerblackbox.modules.ner_training.data_preprocessing.tools.bert_dataset import (
    BertDataset,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.csv_reader import (
    CsvReader,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example_to_tensors import (
    InputExampleToTensors,
)
from nerblackbox.modules.utils.util_functions import get_dataset_path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class DataPreprocessor:
    def __init__(self, tokenizer, do_lower_case, default_logger, max_seq_length=64):
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

    def get_input_examples_train(self, dataset_name, prune_ratio):
        """
        get input examples for TRAIN from csv files
        -------------------------------------------
        :param dataset_name:     [str], e.g. 'suc'
        :param prune_ratio:      [dict], e.g. {'train': 1.0, 'val': 1.0, 'test': 1.0} -- pruning ratio for data
        :return: input_examples: [dict] w/ keys = 'train', 'val', 'test' & values = [list] of [InputExample]
        :return: tag_list:       [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
        """
        if prune_ratio is None:
            prune_ratio = {"train": 1.0, "val": 1.0, "test": 1.0}

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

        return input_examples, csv_reader.tag_list

    def get_input_examples_predict(self, examples):
        """
        get input examples for PREDICT from input argument examples
        -----------------------------------------------------------
        :param examples:         [list] of [str]
        :return: input_examples: [dict] w/ key = 'predict' & value = [list] of [InputExample]
        """
        # create input_examples
        if self.do_lower_case:
            examples = [example.lower() for example in examples]

        input_examples = {
            "predict": [
                InputExample(
                    guid="",
                    text_a=example,
                    tags_a=" ".join(
                        ["O" for _ in range(len(example.split()))]
                    ),  # pseudo tags
                )
                for example in examples
            ]
        }
        # print('TEMP: input_examples:', input_examples)

        return input_examples

    def to_dataloader(self, input_examples, tag_list, batch_size):
        """
        turn input_examples into dataloader
        -----------------------------------
        :param input_examples:
        :param input_examples: [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                         values = [list] of [InputExample]
        :param tag_list:         [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
        :param batch_size:       [int]
        :return: _dataloader:    [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                           values = [torch Dataloader]
        """
        # input_example_to_tensors
        input_example_to_tensors = InputExampleToTensors(
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            tag_tuple=tuple(tag_list),
            default_logger=self.default_logger,
        )

        _dataloader = dict()
        for phase in input_examples.keys():
            # dataloader
            data = BertDataset(
                input_examples[phase], transform=input_example_to_tensors
            )

            if phase == "train":
                sampler = RandomSampler(data)
            elif phase in ["val", "test"]:
                sampler = SequentialSampler(data)
            else:
                sampler = None

            _dataloader[phase] = DataLoader(
                data, sampler=sampler, batch_size=batch_size
            )

        return _dataloader

    ####################################################################################################################
    # HELPER
    ####################################################################################################################
    def _prune_examples(self, list_of_examples, phase, ratio=None):
        """
        prunes list_of_examples by taking only the first examples
        ---------------------------------------------------------
        :param list_of_examples: [list], e.g. of [InputExample]
        :param phase:            [str], 'train' or 'valid'
        :param (Optional) ratio: [float], e.g. 0.5
        :return: [list], e.g. of [InputExample]
        """
        if ratio is None:
            return list_of_examples
        else:
            num_examples_old = len(list_of_examples)
            num_examples_new = int(ratio * float(num_examples_old))
            info = f"> {phase.ljust(5)} data: use {num_examples_new} of {num_examples_old} examples"
            self.default_logger.log_info(info)
            return list_of_examples[:num_examples_new]
