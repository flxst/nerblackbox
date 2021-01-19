from nerblackbox.modules.datasets.formatter.swedish_ner_corpus_formatter import (
    SwedishNerCorpusFormatter,
)
from nerblackbox.modules.datasets.formatter.suc_formatter import SUCFormatter
from nerblackbox.modules.datasets.formatter.conll2003_formatter import (
    CoNLL2003Formatter,
)
from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter


class AutoFormatter:
    @staticmethod
    def for_dataset(ner_dataset: str) -> BaseFormatter:
        if ner_dataset == "swedish_ner_corpus":
            return SwedishNerCorpusFormatter()
        elif ner_dataset == "suc":
            return SUCFormatter()
        elif ner_dataset == "conll2003":
            return CoNLL2003Formatter()
        else:
            raise Exception(f"ner_dataset = {ner_dataset} unknown.")
