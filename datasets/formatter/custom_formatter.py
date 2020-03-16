
from datasets.formatter.swedish_ner_corpus_formatter import SwedishNerCorpusFormatter
from datasets.formatter.suc_formatter import SUCFormatter


class CustomFormatter:

    @staticmethod
    def for_dataset(ner_dataset: str):
        if ner_dataset == 'swedish_ner_corpus':
            _formatter = SwedishNerCorpusFormatter()
        elif ner_dataset == 'SUC':
            _formatter = SUCFormatter()
        else:
            raise Exception(f'ner_dataset = {ner_dataset} unknown.')

        return _formatter
