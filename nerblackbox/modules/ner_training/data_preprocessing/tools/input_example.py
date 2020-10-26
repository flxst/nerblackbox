class InputExample:
    """
    A single training/test example for simple sequence classification
    """

    def __init__(self, guid, text_a, tags_a=None, text_b=None, tags_b=None):
        """
        :param guid:   [int] unique id for input example
        :param text_a: [str] raw (untokenized) text of first sequence.
        :param tags_a: [str] labels of first sequence, separated by whitespace.
        :param text_b: [str, optional] raw (untokenized) text of second sequence.
        :param tags_b: [str, optional] labels of second sequence, separated by whitespace.
        """
        self.guid = guid
        self.text_a = text_a
        self.tags_a = tags_a
        self.text_b = text_b
        self.tags_b = tags_b
