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
        "source_scheme, " "input_sequence, " "target_scheme, " "output_sequence",
        [
            (
                    "plain",
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
                    "bio",
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                    "bio",
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                    "bio",
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                    "bio",
                    ["O", "A", "A", "O", "O", "O", "B", "O"],
                    "bio",
                    None,
            ),
            (
                    "plain",
                    ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                    "bio",
                    None,
            ),
        ],
    )
    def test_convert_scheme(
            self,
            source_scheme: str,
            input_sequence: List[str],
            target_scheme: str,
            output_sequence: List[str],
    ):
        if output_sequence is not None:
            test_output_sequence = Tags(input_sequence).convert_scheme(source_scheme, target_scheme)
            assert (
                    test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                Tags(input_sequence).convert_scheme(source_scheme, target_scheme)


if __name__ == "__main__":
    test_tags = TestTags()
    test_tags.test_convert_scheme()
