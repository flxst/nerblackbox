
import pytest
from typing import List, Dict
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.annotation_tags.input_examples_utils import InputExamplesUtils
from nerblackbox.modules.ner_training.annotation_tags.annotation import Annotation


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestInputExamplesUtils:

    ####################################################################################################################
    @pytest.mark.parametrize(
        "tag_classes, " "input_examples, " "tag_classes_bio, " "input_examples_bio",
        [
            (
                    ["O", "PER", "ORG", "MISC"],
                    {
                        "train": [
                            InputExample(
                                guid="",
                                text="På skidspår.se kan längdskidåkare själva betygsätta förhållandena i spåren .",
                                tags="O MISC O O O O O O O O",
                            ),
                        ],
                        "val": [
                            InputExample(
                                guid="",
                                text="Fastigheten är ett landmärke designad av arkitekten Robert Stern .",
                                tags="O O O O O O O PER PER O",
                            ),
                        ],
                        "test": [
                            InputExample(
                                guid="",
                                text="Apple noteras för 85,62 poäng , vilket är den högsta siffran någonsin i undersökningen .",
                                tags="ORG O O O O O O O O O O O O O O",
                            ),
                        ],
                        "predict": [
                            InputExample(
                                guid="",
                                text="På skidspår.se kan längdskidåkare själva betygsätta förhållandena i spåren .",
                                tags="O MISC O O O O O O O O",
                            ),
                            InputExample(
                                guid="",
                                text="Fastigheten är ett landmärke designad av arkitekten Robert Stern .",
                                tags="O O O O O O O PER PER O",
                            ),
                            InputExample(
                                guid="",
                                text="Apple noteras för 85,62 poäng , vilket är den högsta siffran någonsin i undersökningen .",
                                tags="ORG O O O O O O O O O O O O O O",
                            ),
                        ],
                    },
                    ["O", "B-PER", "B-ORG", "B-MISC", "I-PER", "I-ORG", "I-MISC"],
                    {
                        "train": [
                            InputExample(
                                guid="",
                                text="På skidspår.se kan längdskidåkare själva betygsätta förhållandena i spåren .",
                                tags="O B-MISC O O O O O O O O",
                            ),
                        ],
                        "val": [
                            InputExample(
                                guid="",
                                text="Fastigheten är ett landmärke designad av arkitekten Robert Stern .",
                                tags="O O O O O O O B-PER I-PER O",
                            ),
                        ],
                        "test": [
                            InputExample(
                                guid="",
                                text="Apple noteras för 85,62 poäng , vilket är den högsta siffran någonsin i undersökningen .",
                                tags="B-ORG O O O O O O O O O O O O O O",
                            ),
                        ],

                    }
            ),
        ],
    )
    def tests(
            self,
            tag_classes: List[str],
            input_examples: Dict[str, List[InputExample]],
            tag_classes_bio: List[str],
            input_examples_bio: Dict[str, List[InputExample]],
    ) -> None:
        # b1. convert input_examples to bio
        test_input_examples_bio = InputExamplesUtils.convert_annotation_scheme(
            input_examples=input_examples,
            annotation_scheme_source=Annotation(tag_classes).scheme,
            annotation_scheme_target="bio",
        )
        for phase in ["train", "val", "test"]:
            assert (
                    len(test_input_examples_bio[phase]) == 1
            ), f"ERROR! len(test_input_examples_bio[{phase}]) = {len(test_input_examples_bio[phase])} should be 1."
            assert (
                    test_input_examples_bio[phase][0].text == input_examples_bio[phase][0].text
            ), f"phase = {phase}: test_input_examples_bio.text = {test_input_examples_bio[phase][0].text} != {input_examples_bio[phase][0].text}"
            assert (
                    test_input_examples_bio[phase][0].tags == input_examples_bio[phase][0].tags
            ), f"phase = {phase}: test_input_examples_bio.tags = {test_input_examples_bio[phase][0].tags} != {input_examples_bio[phase][0].tags}"

        # b2. convert input_examples back from bio
        test_input_examples_2 = InputExamplesUtils.convert_annotation_scheme(
            input_examples=test_input_examples_bio,
            annotation_scheme_source=Annotation(tag_classes_bio).scheme,
            annotation_scheme_target="plain",
        )
        for phase in ["train", "val", "test"]:
            assert (
                    len(test_input_examples_2[phase]) == 1
            ), f"ERROR! len(test_input_examples_2[{phase}]) = {len(test_input_examples_2[phase])} should be 1."
            assert (
                    test_input_examples_2[phase][0].text == input_examples[phase][0].text
            ), f"phase = {phase}: test_input_examples_2.text = {test_input_examples_2[phase][0].text} != {input_examples[phase][0].text}"
            assert (
                    test_input_examples_2[phase][0].tags == input_examples[phase][0].tags
            ), f"phase = {phase}: test_input_examples_2.tags = {test_input_examples_2[phase][0].tags} != {input_examples[phase][0].tags}"

        # b3. no_conversion
        test_input_examples_plain = InputExamplesUtils.convert_annotation_scheme(
            input_examples=input_examples,
            annotation_scheme_source=Annotation(tag_classes).scheme,
            annotation_scheme_target="plain",
        )
        for phase in ["train", "val", "test"]:
            assert (
                    len(test_input_examples_plain[phase]) == 1
            ), f"ERROR! len(test_input_examples_plain[{phase}]) = {len(test_input_examples_plain[phase])} should be 1."
            assert (
                    test_input_examples_plain[phase][0].text == input_examples[phase][0].text
            ), f"phase = {phase}: test_input_examples_plain.text = {test_input_examples_plain[phase][0].text} != {input_examples[phase][0].text}"
            assert (
                    test_input_examples_plain[phase][0].tags == input_examples[phase][0].tags
            ), f"phase = {phase}: test_input_examples_plain.tags = {test_input_examples_plain[phase][0].tags} != {input_examples[phase][0].tags}"

        # b4. use unknown parameter
        with pytest.raises(Exception):
            _ = InputExamplesUtils.convert_annotation_scheme(
                input_examples=input_examples,
                annotation_scheme_source=Annotation(tag_classes).scheme,
                annotation_scheme_target="xyz",
            )


if __name__ == "__main__":
    test_input_example_utils = TestInputExamplesUtils()
    test_input_example_utils.tests()
