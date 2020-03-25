
from datasets.formatter.swedish_ner_corpus_formatter import SwedishNerCorpusFormatter
from datasets.formatter.suc_formatter import SUCFormatter
from datasets.formatter.conll2003_formatter import CoNLL2003Formatter


class CustomFormatter:

    @staticmethod
    def for_dataset(ner_dataset: str):
        if ner_dataset == 'swedish_ner_corpus':
            _formatter = SwedishNerCorpusFormatter()
        elif ner_dataset == 'SUC':
            _formatter = SUCFormatter()
        elif ner_dataset == 'conll2003':
            _formatter = CoNLL2003Formatter()
        else:
            raise Exception(f'ner_dataset = {ner_dataset} unknown.')

        return _formatter
