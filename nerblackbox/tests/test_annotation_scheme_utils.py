import pytest
from typing import List
from nerblackbox.modules.ner_training.annotation_scheme.annotation_scheme_utils import AnnotationSchemeUtils


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestAnnotationSchemeUtils:

    ####################################################################################################################
    @pytest.mark.parametrize(
        "tag_list, " "returned_tag_list",
        [
            (
                    ["O", "PER", "ORG", "MISC"],
                    ["O", "PER", "ORG", "MISC"],
            ),
            (
                    ["O", "B-PER", "B-ORG", "B-MISC"],
                    ["O", "B-PER", "B-ORG", "B-MISC", "I-PER", "I-ORG", "I-MISC"],
            ),
        ],
    )
    def test_ensure_completeness_in_case_of_bio_tags(
            self,
            tag_list: List[str],
            returned_tag_list: List[str],
    ) -> None:
        test_returned_tag_list = (
            AnnotationSchemeUtils.ensure_completeness_in_case_of_bio_tags(
                tag_list=tag_list
            )
        )
        assert (
                test_returned_tag_list == returned_tag_list
        ), f"test_returned_tag_list = {test_returned_tag_list} != {returned_tag_list}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "tag_list, " "tag_list_ordered",
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
    def test_order_tag_list(
            self,
            tag_list: List[str],
            tag_list_ordered: List[str],
    ) -> None:
        test_tag_list_ordered = AnnotationSchemeUtils.order_tag_list(tag_list)
        assert (
                test_tag_list_ordered == tag_list_ordered
        ), f"test_tag_list_ordered = {test_tag_list_ordered} != {tag_list_ordered}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "tag_list_bio, " "tag_list",
        [
            (
                    ["O", "B-MISC", "B-ORG", "B-PER", "I-MISC", "I-ORG", "I-PER"],
                    ["O", "MISC", "ORG", "PER"],
            ),
            (  # if applied to plain tag_list, nothing happens
                    ["O", "MISC", "ORG", "PER"],
                    ["O", "MISC", "ORG", "PER"],
            ),
        ],
    )
    def test_convert_tag_list_bio2plain(
            self,
            tag_list_bio: List[str],
            tag_list: List[str],
    ) -> None:
        test_tag_list = AnnotationSchemeUtils.convert_tag_list_bio2plain(tag_list_bio)
        assert (
                test_tag_list == tag_list
        ), f"test_tag_list_ordered = {test_tag_list} != {tag_list}"

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    @pytest.mark.parametrize(
        "input_sequence, " "convert_to_bio, " "output_sequence",
        [
            (
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
                    True,
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                    False,
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
                    False,
                    None,
            ),
            (
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                    True,
                    None,
            ),
        ],
    )
    def test_convert2bio(
            self,
            input_sequence: List[str],
            convert_to_bio: bool,
            output_sequence: List[str],
    ):
        if output_sequence is not None:
            test_output_sequence = AnnotationSchemeUtils.convert2bio(input_sequence, convert_to_bio)
            assert (
                    test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                AnnotationSchemeUtils.convert2bio(input_sequence, convert_to_bio)

    @pytest.mark.parametrize(
        "input_sequence, " "convert_to_plain, " "output_sequence",
        [
            (
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                    True,
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
            ),
            (
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                    False,
                    None,
            ),
            (
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
                    False,
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
            ),
        ],
    )
    def test_convert2plain(
            self,
            input_sequence: List[str],
            convert_to_plain: bool,
            output_sequence: List[str],
    ):
        if output_sequence is not None:
            test_output_sequence = AnnotationSchemeUtils.convert2plain(input_sequence, convert_to_plain)
            assert (
                    test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                AnnotationSchemeUtils.convert2plain(input_sequence, convert_to_plain)


if __name__ == "__main__":
    test_annotation_scheme_utils = TestAnnotationSchemeUtils()
    test_annotation_scheme_utils.test_ensure_completeness_in_case_of_bio_tags()
    test_annotation_scheme_utils.test_order_tag_list()
    test_annotation_scheme_utils.test_convert_tag_list_bio2plain()
    test_annotation_scheme_utils.test_convert2bio()
    test_annotation_scheme_utils.test_convert2plain()
