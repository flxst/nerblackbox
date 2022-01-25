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
            # 1. plain -> plain
            (
                "plain",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
                "plain",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
            ),
            # 2. plain -> bio
            (
                "plain",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            # 3. plain -> bilou
            (
                "plain",
                ["O", "A", "A", "A", "O", "O", "O", "B", "O"],
                "bilou",
                ["O", "B-A", "I-A", "L-A", "O", "O", "O", "U-B", "O"],
            ),
            # 4. bio -> plain
            (
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "plain",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
            ),
            # 5. bio -> bio
            (
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                "bio",
                ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                "bio",
                ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                # ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                "bio",
                ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                "bio",
                ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                # ["B-B", "B-A", "B-A", "O", "O", "B-A", "B-B", "O"],
            ),
            # 6. bio -> bilou
            (
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
            ),
            # 7. bilou -> plain
            (
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
                "plain",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
            ),
            # 8. bilou -> bio
            (
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            # 9. bilou -> bilou
            (
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
            ),
            (
                "bilou",
                ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                "bilou",
                ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                # ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
            ),
            (
                "bilou",
                ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                "bilou",
                ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                # ["U-B", "U-A", "U-A", "O", "O", "U-A", "U-B", "O"],
            ),
            (
                "bilou",
                ["B-A", "I-A", "I-A", "O", "O", "I-A", "I-B", "O"],
                "bilou",
                ["B-A", "I-A", "I-A", "O", "O", "I-A", "I-B", "O"],
                # ["B-A", "I-A", "L-A", "O", "O", "U-A", "U-B", "O"],
            ),
            # 10. unknown input scheme
            (
                "xyz",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "bio",
                None,
            ),
            # 11. wrong input scheme
            (
                "plain",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "bio",
                None,
            ),
            (
                "bio",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
                "bio",
                None,
            ),
            (
                "bilou",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
                "bio",
                None,
            ),
            # 12. incorrect tag format
            (
                "bio",
                ["O", "B-A", "L-A", "O", "O", "O", "B-B", "O"],
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
            test_output_sequence = Tags(input_sequence).convert_scheme(
                source_scheme, target_scheme
            )
            assert (
                test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                Tags(input_sequence).convert_scheme(source_scheme, target_scheme)

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    @pytest.mark.parametrize(
        "source_scheme, " "input_sequence, " "target_scheme, " "output_sequence",
        [
            # 5. bio -> bio
            (
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                "bio",
                ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                "bio",
                # ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
            ),
            (
                "bio",
                ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                "bio",
                # ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                ["B-B", "B-A", "B-A", "O", "O", "B-A", "B-B", "O"],
            ),
            # 9. bilou -> bilou
            (
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
                "bilou",
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
            ),
            (
                "bilou",
                ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                "bilou",
                # ["O", "I-A", "I-A", "O", "O", "O", "I-B", "O"],
                ["O", "B-A", "L-A", "O", "O", "O", "U-B", "O"],
            ),
            (
                "bilou",
                ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                "bilou",
                # ["B-B", "I-A", "B-A", "O", "O", "I-A", "I-B", "O"],
                ["U-B", "U-A", "U-A", "O", "O", "U-A", "U-B", "O"],
            ),
            (
                "bilou",
                ["B-A", "I-A", "I-A", "O", "O", "I-A", "I-B", "O"],
                "bilou",
                # ["B-A", "I-A", "I-A", "O", "O", "I-A", "I-B", "O"],
                ["B-A", "I-A", "L-A", "O", "O", "U-A", "U-B", "O"],
            ),
            # 11. wrong input scheme
            (
                "bio",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
                "bio",
                None,
            ),
            (
                "bilou",
                ["O", "A", "A", "O", "O", "O", "B", "O"],
                "bio",
                None,
            ),
            # 12. incorrect tag format
            (
                "bio",
                ["O", "B-A", "L-A", "O", "O", "O", "B-B", "O"],
                "bio",
                None,
            ),
            # 13. unknown output scheme
            (
                "bio",
                ["O", "B-A", "I-A", "O", "O", "O", "B-B", "O"],
                "xyz",
                None,
            ),
        ],
    )
    def test_restore_annotation_scheme_consistency(
        self,
        source_scheme: str,
        input_sequence: List[str],
        target_scheme: str,
        output_sequence: List[str],
    ):
        if output_sequence is not None:
            test_output_sequence = Tags(
                input_sequence
            ).restore_annotation_scheme_consistency(target_scheme)[0]
            assert (
                test_output_sequence == output_sequence
            ), f"{test_output_sequence} != {output_sequence}"
        else:
            with pytest.raises(Exception):
                Tags(input_sequence).restore_annotation_scheme_consistency(
                    target_scheme
                )
