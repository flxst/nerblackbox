
import pytest
import numpy as np
from typing import Dict
from nerblackbox.modules.ner_training.ner_model_evaluation import NerModelEvaluation
from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics


class TestNerModelEvaluation:

    ner_model_evaluation = NerModelEvaluation(
        current_epoch=2,
        tag_list=["O", "PER", "ORG"],
        dataset_tags=["O", "PER", "ORG", "MISC"],
        default_logger=None,
        logged_metrics=LoggedMetrics(),
    )

    ####################################################################################################################
    # TEST #############################################################################################################
    ####################################################################################################################
    @pytest.mark.parametrize(
        "_np_tag_ids, "
        "_np_logits, "
        "true_flat, "
        "pred_flat",
        [
            (
                np.array([[0, 1],
                          [1, 0]]),
                np.array([[[0.8, 0.2], [0.8, 0.2]],
                          [[0.3, 0.7], [0.3, 0.7]]]),
                np.array([0, 1, 1, 0]),
                np.array([0, 0, 1, 1]),
            ),
        ]
    )
    def test_reduce_and_flatten(
            self,
            _np_tag_ids: np.array,
            _np_logits: np.array,
            true_flat: np.array,
            pred_flat: np.array,
    ) -> None:
        test_true_flat, test_pred_flat = self.ner_model_evaluation._reduce_and_flatten(_np_tag_ids, _np_logits)
        assert list(test_true_flat) == list(true_flat), f"true_flat: test = {test_true_flat} != {true_flat} = true"
        assert list(test_pred_flat) == list(pred_flat), f"pred_flat: test = {test_true_flat} != {pred_flat} = true"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_np_tag_ids, "
        "tags",
        [
            (
                    np.array([0, 1, 2, 1]),
                    np.array(["O", "PER", "ORG", "PER"]),
            ),
            (
                    np.array([-100, 1, 2, -100]),
                    np.array(["[S]", "PER", "ORG", "[S]"]),
            ),
        ]
    )
    def test_convert_tag_ids_to_tags(
            self,
            _np_tag_ids: np.array,
            tags: np.array,
    ) -> None:
        test_tags = self.ner_model_evaluation._convert_tag_ids_to_tags(_np_tag_ids)
        assert list(test_tags) == list(tags), f"test = {test_tags} != {tags} = true"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_tags, "
        "tags_new",
        [
            (
                    {
                        "true": np.array(["[S]", "PER", "ORG", "[S]"]),
                        "pred": np.array(["PER", "ORG",   "O",   "O"]),
                    },
                    {
                        "true": np.array(["PER", "ORG"]),
                        "pred": np.array(["ORG",   "O"]),
                    },

            ),
        ]
    )
    def test_get_rid_of_special_tag_occurrences(
            self,
            _tags: Dict[str, np.array],
            tags_new: Dict[str, np.array],
    ) -> None:
        test_tags_new = self.ner_model_evaluation._get_rid_of_special_tag_occurrences(_tags)
        for k in tags_new.keys():
            assert list(test_tags_new[k]) == list(tags_new[k]), \
                f"key = {k}: test = {test_tags_new[k]} != {tags_new[k]} = true"


if __name__ == "__main__":
    test = TestNerModelEvaluation()
    test.test_reduce_and_flatten()
    test.test_convert_tag_ids_to_tags()
    test.test_get_rid_of_special_tag_occurrences()
