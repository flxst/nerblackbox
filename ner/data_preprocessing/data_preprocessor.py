
from ner.data_preprocessing.tools.bert_dataset import BertDataset
from ner.data_preprocessing.tools.csv_reader import CsvReader
from ner.data_preprocessing.tools.input_example_to_tensors import InputExampleToTensors
from ner.utils.util_functions import get_dataset_path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class DataPreprocessor:

    def __init__(self,
                 dataset_name,
                 tokenizer,
                 batch_size,
                 do_lower_case,
                 default_logger,
                 max_seq_length=64,
                 prune_ratio={'train': 1.0, 'val': 1.0, 'test': 1.0}):
        """
        :param dataset_name:   [str], e.g. 'suc'
        :param tokenizer:      [transformers Tokenizer]
        :param batch_size:     [int], e.g. 16
        :param do_lower_case:  [bool] if True, make text data lowercase
        :param default_logger: [DefaultLogger]
        :param max_seq_length: [int], e.g. 64
        :param prune_ratio:    [dict], e.g. {'train': 1.0, 'val': 1.0, 'test': 1.0} -- pruning ratio for data
        """
        self.dataset_path = get_dataset_path(dataset_name)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.do_lower_case = do_lower_case
        self.default_logger = default_logger
        self.max_seq_length = max_seq_length
        self.prune_ratio = prune_ratio

    def preprocess(self):
        """
        preprocess dataset
        :return: dataloader [dict] w/ keys = 'train', 'valid' & values = [pytorch DataLoader]
        :return: tag_list   [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
        """
        # csv_reader
        csv_reader = CsvReader(self.dataset_path,
                               self.tokenizer,
                               do_lower_case=self.do_lower_case,   # can be True (applies .lower()) !!
                               default_logger=self.default_logger)

        # input_example_to_tensors
        input_example_to_tensors = InputExampleToTensors(self.tokenizer,
                                                         max_seq_length=self.max_seq_length,
                                                         tag_tuple=tuple(csv_reader.tag_list),
                                                         default_logger=self.default_logger)

        dataloader = dict()
        for phase in ['train', 'val', 'test']:
            # train data
            input_examples_all = csv_reader.get_input_examples(phase)
            input_examples = self.prune_examples(input_examples_all, phase, ratio=self.prune_ratio[phase])

            # dataloader
            data = BertDataset(input_examples,
                               transform=input_example_to_tensors)

            if phase == 'train':
                sampler = RandomSampler(data)
            else:
                sampler = SequentialSampler(data)

            dataloader[phase] = DataLoader(data,
                                           sampler=sampler,
                                           batch_size=self.batch_size)

        return dataloader, csv_reader.tag_list

    def prune_examples(self, list_of_examples, phase, ratio=None):
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
            num_examples_new = int(ratio*float(num_examples_old))
            info = f'> {phase.ljust(5)} data: use {num_examples_new} of {num_examples_old} examples'
            self.default_logger.log_info(info)
            return list_of_examples[:num_examples_new]
