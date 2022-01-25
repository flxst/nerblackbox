from nerblackbox.modules.datasets.formatter.swedish_ner_corpus_formatter import (
    SwedishNerCorpusFormatter,
)
from nerblackbox.modules.datasets.formatter.sic_formatter import SICFormatter
from nerblackbox.modules.datasets.formatter.suc_formatter import SUCFormatter
from nerblackbox.modules.datasets.formatter.conll2003_formatter import (
    CoNLL2003Formatter,
)
from nerblackbox.modules.datasets.formatter.base_formatter import BaseFormatter
from nerblackbox.modules.datasets.formatter.swe_nerc_formatter import SweNercFormatter
from nerblackbox.modules.datasets.formatter.huggingface_datasets_formatter import (
    HuggingfaceDatasetsFormatter,
)


class AutoFormatter:
    @staticmethod
    def for_dataset(ner_dataset: str) -> BaseFormatter:
        """
        Args:
            ner_dataset: e.g. "conll2003"

        Returns:
            formatter: e.g. CoNLL2003Formatter
        """
        if ner_dataset == "swedish_ner_corpus":
            return SwedishNerCorpusFormatter()
        elif ner_dataset == "conll2003":
            return CoNLL2003Formatter()
        elif ner_dataset == "sic":
            return SICFormatter()
        elif ner_dataset == "suc":
            return SUCFormatter()
        elif ner_dataset == "swe_nerc":
            return SweNercFormatter()
        else:  # huggingface datasets
            if HuggingfaceDatasetsFormatter.check_existence(ner_dataset):
                print(f"> ner_dataset = {ner_dataset} found in huggingface datasets")
            else:
                raise Exception(f"ner_dataset = {ner_dataset} unknown.")

            if HuggingfaceDatasetsFormatter.check_compatibility(ner_dataset):
                print(f"> ner_dataset = {ner_dataset} contains train/val/test splits")
            else:
                raise Exception(
                    f"ner_dataset = {ner_dataset} does not contain train/val/test splits."
                )

            if HuggingfaceDatasetsFormatter.check_implementation(ner_dataset):
                print(f"> ner_dataset = {ner_dataset} can be parsed")
            else:
                raise Exception(f"ner_dataset = {ner_dataset} can not be parsed.")

            return HuggingfaceDatasetsFormatter(ner_dataset)
