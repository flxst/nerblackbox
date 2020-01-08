
import csv


########################################################################################################################
# from processors.py
########################################################################################################################
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, *args, **kwargs):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_label_list(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, delimiter='\t', quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
