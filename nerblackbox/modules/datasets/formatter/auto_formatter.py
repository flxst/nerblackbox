from nerblackbox.modules.datasets.formatter.swedish_ner_corpus_formatter import (
    SwedishNerCorpusFormatter,
)
from nerblackbox.modules.datasets.formatter.sic_formatter import SICFormatter
from nerblackbox.modules.datasets.formatter.suc_formatter import SUCFormatter
from nerblackbox.modules.datasets.formatter.sucx_formatter import SUCXFormatter
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
    def for_dataset(ner_dataset: str, ner_dataset_subset: str = "", verbose: bool = False) -> BaseFormatter:
        """
        Args:
            ner_dataset: e.g. "conll2003"
            ner_dataset_subset: e.g. "simple_cased"
            verbose: output

        Returns:
            formatter: e.g. CoNLL2003Formatter
        """
        if ner_dataset == "swedish_ner_corpus":
            return SwedishNerCorpusFormatter()
        elif ner_dataset == "conll2003_from_source":
            return CoNLL2003Formatter()
        elif ner_dataset == "sic":
            return SICFormatter()
        elif ner_dataset == "suc":
            return SUCFormatter()
        elif ner_dataset == "sucx":
            assert len(
                ner_dataset_subset
            ), f"ERROR! for sucx, a subset needs to be specified."
            return SUCXFormatter(ner_dataset_subset)
        elif ner_dataset == "swe_nerc":
            return SweNercFormatter()
        else:  # huggingface datasets
            existence, error_msg = HuggingfaceDatasetsFormatter.check_existence(
                ner_dataset, ner_dataset_subset
            )
            if existence:
                if verbose:
                    print(f"> ner_dataset = {ner_dataset} found in huggingface datasets")
            else:
                raise Exception(error_msg)

            compatibility, error_msg = HuggingfaceDatasetsFormatter.check_compatibility(
                ner_dataset, ner_dataset_subset
            )
            if compatibility:
                if verbose:
                    print(f"> ner_dataset = {ner_dataset} contains train/val/test splits")
            else:
                raise Exception(error_msg)

            (
                implementation,
                error_msg,
            ) = HuggingfaceDatasetsFormatter.check_implementation(
                ner_dataset, ner_dataset_subset
            )
            if implementation:
                if verbose:
                    print(f"> ner_dataset = {ner_dataset} can be parsed")
            else:
                raise Exception(error_msg)

            return HuggingfaceDatasetsFormatter(ner_dataset, ner_dataset_subset)
