import pytest
from typing import List
from nerblackbox.modules.ner_training.annotation_tags.tags import Tags


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestTags:

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
            test_output_sequence = Tags(input_sequence).convert2bio(convert_to_bio)
            assert (
                    test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                Tags(input_sequence).convert2bio(convert_to_bio)

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
            test_output_sequence = Tags(input_sequence).convert2plain(convert_to_plain)
            assert (
                    test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                Tags(input_sequence).convert2plain(convert_to_plain)


if __name__ == "__main__":
    test_tags = TestTags()
    test_tags.test_convert2bio()
    test_tags.test_convert2plain()
