
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, labels_a=None, text_b=None, labels_b=None):
        """Constructs an InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            labels_a: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels_b: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels_a = labels_a
        self.text_b = text_b
        self.labels_b = labels_b
