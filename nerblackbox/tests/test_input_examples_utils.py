import pytest
from typing import List, Dict
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.annotation_tags.input_examples_utils import (
    InputExamplesUtils,
)
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
                },
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
        def _test(
            _test_input_examples: Dict[str, List[InputExample]],
            _input_examples: Dict[str, List[InputExample]],
            _test_name: str,
        ):
            """
            test _test_input_examples against ground truth _input_examples

            Args:
                _test_input_examples: input examples to be tested
                _input_examples:      input examples to be tested against
                _test_name:           _test_input_examples as string
            """
            for phase in ["train", "val", "test"]:
                assert (
                    len(_test_input_examples[phase]) == 1
                ), f"ERROR! len({_test_name}[{phase}]) = {len(_test_input_examples[phase])} should be 1."
                assert (
                    _test_input_examples[phase][0].text
                    == _input_examples[phase][0].text
                ), f"phase = {phase}: {_test_name}.text = {_test_input_examples[phase][0].text} != {_input_examples[phase][0].text}"
                assert (
                    _test_input_examples[phase][0].tags
                    == _input_examples[phase][0].tags
                ), f"phase = {phase}: {_test_name}.tags = {_test_input_examples[phase][0].tags} != {_input_examples[phase][0].tags}"

        # b1. plain -> bio
        test_input_examples_plain2bio = InputExamplesUtils.convert_annotation_scheme(
            input_examples=input_examples,
            annotation_scheme_source=Annotation(tag_classes).scheme,
            annotation_scheme_target="bio",
        )
        _test(
            test_input_examples_plain2bio,
            input_examples_bio,
            "test_input_examples_plain2bio",
        )

        # b2. bio -> plain
        test_input_examples_bio2plain = InputExamplesUtils.convert_annotation_scheme(
            input_examples=input_examples_bio,
            annotation_scheme_source=Annotation(tag_classes_bio).scheme,
            annotation_scheme_target="plain",
        )
        _test(
            test_input_examples_bio2plain,
            input_examples,
            "test_input_examples_bio2plain",
        )

        # b3. plain -> plain
        test_input_examples_plain = InputExamplesUtils.convert_annotation_scheme(
            input_examples=input_examples,
            annotation_scheme_source=Annotation(tag_classes).scheme,
            annotation_scheme_target="plain",
        )
        _test(test_input_examples_plain, input_examples, "test_input_examples_plain")

        # b4. bio -> bio
        test_input_examples_bio = InputExamplesUtils.convert_annotation_scheme(
            input_examples=input_examples_bio,
            annotation_scheme_source=Annotation(tag_classes_bio).scheme,
            annotation_scheme_target="bio",
        )
        _test(test_input_examples_bio, input_examples_bio, "test_input_examples_bio")

        # b5. use unknown parameter
        with pytest.raises(Exception):
            _ = InputExamplesUtils.convert_annotation_scheme(
                input_examples=input_examples,
                annotation_scheme_source=Annotation(tag_classes).scheme,
                annotation_scheme_target="xyz",
            )
