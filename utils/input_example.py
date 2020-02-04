
class InputExample:
    """
    A single training/test example for simple sequence classification
    """

    def __init__(self, guid, text_a, labels_a=None, text_b=None, labels_b=None):
        """
        :param guid:     [int] unique id for input example
        :param text_a:   [str] raw (untokenized) text of first sequence.
        :param labels_a: [str] labels of first sequence, separated by whitespace.
        :param text_b:   [str, optional] raw (untokenized) text of second sequence.
        :param labels_b: [str, optional] labels of second sequence, separated by whitespace.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels_a = labels_a
        self.text_b = text_b
        self.labels_b = labels_b
