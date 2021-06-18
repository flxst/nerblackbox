class InputExample:
    """
    A single training/test example for simple sequence classification
    """

    def __init__(self, guid: str, text: str, tags: str = ""):
        """
        :param guid: [str] unique id for input example
        :param text: [str] raw (untokenized) text of first sequence.
        :param tags: [str] labels of first sequence, separated by whitespace.
        """
        self.guid = guid
        self.text = text
        self.tags = tags
