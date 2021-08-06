
from typing import Dict
from copy import deepcopy
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    InputExamples,
)
from nerblackbox.modules.ner_training.annotation_tags.tags import (
    Tags
)


class InputExamplesUtils:

    ####################################################################################################################
    # 1. from data preprocessing
    ####################################################################################################################
    @classmethod
    def convert_annotation_scheme(
            cls,
            input_examples: Dict[str, InputExamples],
            annotation_scheme_source: str,
            annotation_scheme_target: str) -> Dict[str, InputExamples]:
        """
        convert input_examples from annotation_scheme_source to annotation_scheme_target

        Args:
            input_examples: [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                      values = [list] of [InputExample]
            annotation_scheme_source: [str], e.g. plain / bio
            annotation_scheme_target: [str], e.g. bio / plain

        Returns:
            input_examples_converted: [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                                values = [list] of [InputExample]
        """
        if annotation_scheme_source == annotation_scheme_target:
            return input_examples
        else:
            input_examples_converted = deepcopy(input_examples)
            for key in input_examples_converted.keys():
                for input_example_converted in input_examples_converted[key]:
                    tags = Tags(input_example_converted.tags.split())
                    if annotation_scheme_source == "plain" and annotation_scheme_target == "bio":
                        input_example_converted.tags = " ".join(tags.convert2bio())
                    elif annotation_scheme_source == "bio" and annotation_scheme_target == "plain":
                        input_example_converted.tags = " ".join(tags.convert2plain())
                    else:
                        raise Exception(f"annotation_scheme_source = {annotation_scheme_source} and "
                                        f"annotation_scheme_target = {annotation_scheme_target} not implemented.")

        return input_examples_converted
