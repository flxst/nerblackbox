
import pytest
from typing import List, Optional
from nerblackbox.modules.ner_training.annotation_tags.annotation import Annotation


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestAnnotation:

    ####################################################################################################################
    @pytest.mark.parametrize(
        "classes, " "processed_classes",
        [
            (
                    ["O", "PER", "ORG", "MISC"],
                    ["O", "MISC", "ORG", "PER"],
            ),
            (
                    ["O", "B-PER", "B-ORG", "B-MISC"],
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
            ),
        ],
    )
    def test_ensure_completeness_in_case_of_bio_tags(
            self,
            classes: List[str],
            processed_classes: List[str],
    ) -> None:
        annotation = Annotation(classes)
        assert (
                annotation.classes == processed_classes
        ), f"annotation.classes = {annotation.classes} != {processed_classes}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "classes, " "processed_classes",
        [
            (
                    ["O", "PER", "ORG", "MISC"],
                    ["O", "MISC", "ORG", "PER"],
            ),
            (
                    ["PER", "ORG", "O", "MISC"],
                    ["O", "MISC", "ORG", "PER"],
            ),
            (
                    ["O", "B-PER", "I-MISC", "B-ORG", "I-PER", "B-MISC", "I-ORG"],
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
            ),
        ],
    )
    def test_sort(
            self,
            classes: List[str],
            processed_classes: List[str],
    ) -> None:
        annotation = Annotation(classes)
        assert (
                annotation.classes == processed_classes
        ), f"annotation.classes = {annotation.classes} != {processed_classes}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "classes, " "annotation_scheme",
        [
            (
                    ["O", "MISC", "ORG", "PER"],
                    "plain",
            ),
            (
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
                    "bio",
            ),
        ],
    )
    def test_determine_annotation_scheme(
            self,
            classes: List[str],
            annotation_scheme: str,
    ) -> None:
        annotation = Annotation(classes)
        assert annotation.scheme == annotation_scheme, \
            f"ERROR! annotation.scheme = {annotation.scheme} != {annotation_scheme}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "new_scheme, " "old_classes, " "new_classes",
        [
            (
                    "plain",
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
                    ["O", "MISC", "ORG", "PER"],
            ),
            (
                    "plain",
                    ["O", "MISC", "ORG", "PER"],
                    ["O", "MISC", "ORG", "PER"],
            ),
            (
                    "bio",
                    ["O", "MISC", "ORG", "PER"],
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
            ),
            (
                    "bio",
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
            ),
            (
                    "xyz",
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
                    None,
            ),
        ],
    )
    def test_change_scheme(
            self,
            new_scheme: str,
            old_classes: List[str],
            new_classes: Optional[List[str]],
    ) -> None:
        if new_classes is None:
            with pytest.raises(Exception):
                _ = Annotation(old_classes).change_scheme(new_scheme=new_scheme)
        else:
            annotation = Annotation(old_classes).change_scheme(new_scheme=new_scheme)
            assert annotation.scheme == new_scheme, \
                f"ERROR! annotation.scheme = {annotation.scheme} != {new_scheme}"
            assert (
                    annotation.classes == new_classes
            ), f"annotation.classes = {annotation.classes} != {new_classes}"


if __name__ == "__main__":
    test_annotation = TestAnnotation()
    test_annotation.test_ensure_completeness_in_case_of_bio_tags()
    test_annotation.test_sort()
    test_annotation.test_determine_annotation_scheme()
    test_annotation.test_change_scheme()
