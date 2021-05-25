
import pytest
import numpy as np
import torch
from typing import Dict, List, Union
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
        "outputs, "
        "np_batch, "
        "np_epoch",
        [
            (
                [
                    [
                        torch.tensor([1.0,
                                      2.0]),
                        torch.tensor([3,
                                      4]),
                        torch.tensor([[9.0, 1.2, 3.3],
                                      [0.7, 0.6, 5.6]]),
                    ],
                    [
                        torch.tensor([2.0,
                                      3.0]),
                        torch.tensor([5,
                                      6]),
                        torch.tensor([[0.9, 2.1, 3.3],
                                      [7.0, 6.0, 6.5]]),
                    ],
                ],
                {
                    "loss": [
                        np.array([1.0,
                                  2.0], dtype=np.float32),
                        np.array([2.0,
                                  3.0], dtype=np.float32),
                    ],
                    "tag_ids": [
                        np.array([3,
                                  4], dtype=np.float32),
                        np.array([5,
                                  6], dtype=np.float32),
                    ],
                    "logits": [
                        np.array([[9.0, 1.2, 3.3],
                                  [0.7, 0.6, 5.6]], dtype=np.float32),
                        np.array([[0.9, 2.1, 3.3],
                                  [7.0, 6.0, 6.5]], dtype=np.float32),
                    ],
                },
                {
                    "loss": 2.0,
                    "tag_ids": np.array([3,
                                         4,
                                         5,
                                         6], dtype=np.float32),
                    "logits": np.array([[9.0, 1.2, 3.3],
                                        [0.7, 0.6, 5.6],
                                        [0.9, 2.1, 3.3],
                                        [7.0, 6.0, 6.5]], dtype=np.float32),
                },
            ),
        ]
    )
    def test_convert_and_combine(
            self,
            outputs: List[List[torch.tensor]],
            np_batch: Dict[str, List[np.array]],
            np_epoch: Dict[str, Union[np.number, np.array]],
    ) -> None:
        # 1. output -> np_batch
        test_np_batch = self.ner_model_evaluation._convert_output_to_np_batch(outputs)
        for key in np_batch.keys():
            for _test, _true in zip(test_np_batch[key], np_batch[key]):
                assert np.array_equal(_test, _true), \
                    f"test_np_batch[{key}] = {_test} != {_true}"

        # 2. np_batch -> np_epoch
        test_np_epoch = self.ner_model_evaluation._combine_np_batch_to_np_epoch(np_batch)
        for key in np_epoch.keys():
            if key == "loss":
                assert test_np_epoch[key] == np_epoch[key], \
                    f"test_np_epoch[{key}] = {test_np_epoch[key]} != {np_epoch[key]}"
            else:
                for _test, _true in zip(test_np_epoch[key], np_epoch[key]):
                    assert np.array_equal(_test, _true), \
                        f"test_np_epoch[{key}] = {_test} != {_true}"

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

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_tag_subset, "
        "_filtered_tags",
        [
            ("all", ["O", "PER", "ORG"]),
            ("fil", ["PER", "ORG"]),
            ("chk", ["PER", "ORG"]),
            ("O",   ["O"]),
            ("PER", ["PER"]),
            ("ORG", ["ORG"]),
        ]
    )
    def test_get_filtered_tags(
            self,
            _tag_subset: str,
            _filtered_tags: List[str],
    ) -> None:
        test_filtered_tags = self.ner_model_evaluation._get_filtered_tags(_tag_subset)
        assert test_filtered_tags == _filtered_tags, f"test_filtered_tags = {test_filtered_tags} != {_filtered_tags}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_tags, "
        "_phase, "
        "tag_subset, "
        "metrics",
        [
            (
                {
                    "true": np.array(["O", "PER", "ORG", "PER"]),
                    "pred": np.array(["O", "PER", "ORG", "PER"]),
                },
                "val",
                "fil",
                {
                    "fil_recall_micro": 1.0,
                    "fil_recall_macro": 1.0,
                    "fil_precision_micro": 1.0,
                    "fil_precision_macro": 1.0,
                    "fil_f1_micro": 1.0,
                    "fil_f1_macro": 1.0,
                }
            ),
        ]
    )
    def test_compute_metrics_for_tags_subset(
            self,
            _tags: Dict[str, np.array],
            _phase: str,
            tag_subset: str,
            metrics: Dict[str, float],
    ) -> None:
        test_metrics = self.ner_model_evaluation._compute_metrics_for_tags_subset(_tags, _phase, tag_subset)
        for k, v in test_metrics.items():
            assert test_metrics[k] == metrics[k], \
                f"test_metrics[{k}] = {test_metrics[k]} != {metrics[k]}"


if __name__ == "__main__":
    test = TestNerModelEvaluation()
    test.test_convert_and_combine()
    test.test_reduce_and_flatten()
    test.test_convert_tag_ids_to_tags()
    test.test_get_rid_of_special_tag_occurrences()
    test.test_get_filtered_tags()
    test.test_compute_metrics_for_tags_subset()