
from ner.data_preprocessing.tools.bert_dataset import BertDataset
from ner.data_preprocessing.tools.ner_processor import NerProcessor
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
                 prune_ratio=(1.0, 1.0)):
        """
        :param dataset_name:   [str], e.g. 'SUC'
        :param tokenizer:      [transformers Tokenizer]
        :param batch_size:     [int], e.g. 16
        :param do_lower_case:  [bool] if True, make text data lowercase
        :param default_logger: [DefaultLogger]
        :param max_seq_length: [int], e.g. 64
        :param prune_ratio:    [tuple], e.g. (1.0, 1.0) -- pruning ratio for train & valid data
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
        input_examples = dict()
        data = dict()
        dataloader = dict()

        # processor
        processor = NerProcessor(self.dataset_path,
                                 self.tokenizer,
                                 do_lower_case=self.do_lower_case)  # can be True (applies .lower()) !!
        tag_list = processor.get_tag_list()

        # train data
        input_examples_train_all = processor.get_input_examples('train')
        input_examples['train'] = self.prune_examples(input_examples_train_all, 'train', ratio=self.prune_ratio[0])

        # validation data
        input_examples_valid_all = processor.get_input_examples('valid')
        input_examples['valid'] = self.prune_examples(input_examples_valid_all, 'valid', ratio=self.prune_ratio[1])

        # input_examples_to_tensors
        input_examples_to_tensors = InputExampleToTensors(self.tokenizer,
                                                          max_seq_length=self.max_seq_length,
                                                          tag_tuple=tuple(tag_list))

        # dataloader
        data['train'] = BertDataset(input_examples['train'],
                                    transform=input_examples_to_tensors)
        dataloader['train'] = DataLoader(data['train'],
                                         sampler=RandomSampler(data['train']),
                                         batch_size=self.batch_size)

        data['valid'] = BertDataset(input_examples['valid'],
                                    transform=input_examples_to_tensors)
        dataloader['valid'] = DataLoader(data['valid'],
                                         sampler=SequentialSampler(data['valid']),
                                         batch_size=self.batch_size)

        return dataloader, tag_list

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
            self.default_logger.log_info(f'> {phase} data: use {num_examples_new} of {num_examples_old} examples')
            return list_of_examples[:num_examples_new]
