from torch.utils.data import Dataset


class BertDataset(Dataset):
    """ Bert Dataset with features created when requested """

    def __init__(self, samples, transform, max_len=None):
        """
        Args
            samples: List of InputExample instances.
            transform (callable): Transform to be applied on a list of InputExamples
        """
        self.samples = samples
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return self.max_len if self.max_len else len(self.samples)

    def __getitem__(self, index):
        """
        apply self.transform on single sample w/ index = index
        ------------------------------------------------------
        :param index: [int]
        :return: transformed sample
        """
        return self.transform(self.samples[index])
