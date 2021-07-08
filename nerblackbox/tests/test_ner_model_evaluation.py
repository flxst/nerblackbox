import pytest
import numpy as np
import torch
from typing import Dict, List, Union, Optional, Any
from nerblackbox.modules.ner_training.ner_model_evaluation import NerModelEvaluation
from nerblackbox.modules.ner_training.metrics.logged_metrics import LoggedMetrics
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.tests.test_utils import pytest_approx


class TestNerModelEvaluation:

    ner_model_evaluation = NerModelEvaluation(
        current_epoch=2,
        tag_list=["O", "ORG", "PER"],
        annotation_scheme="plain",
        default_logger=PseudoDefaultLogger(),
        logged_metrics=LoggedMetrics(),
    )
    ner_model_evaluation_bio = NerModelEvaluation(
        current_epoch=2,
        tag_list=["O", "B-ORG", "B-PER", "I-ORG", "I-PER"],
        annotation_scheme="bio",
        default_logger=PseudoDefaultLogger(),
        logged_metrics=LoggedMetrics(),
    )

    ####################################################################################################################
    # TEST #############################################################################################################
    ####################################################################################################################
    @pytest.mark.parametrize(
        "outputs, " "np_batch, " "np_epoch",
        [
            (
                [
                    [
                        torch.tensor([1.0, 2.0]),
                        torch.tensor([3, 4]),
                        torch.tensor([[9.0, 1.2, 3.3], [0.7, 0.6, 5.6]]),
                    ],
                    [
                        torch.tensor([2.0, 3.0]),
                        torch.tensor([5, 6]),
                        torch.tensor([[0.9, 2.1, 3.3], [7.0, 6.0, 6.5]]),
                    ],
                ],
                {
                    "loss": [
                        np.array([1.0, 2.0], dtype=np.float32),
                        np.array([2.0, 3.0], dtype=np.float32),
                    ],
                    "tag_ids": [
                        np.array([3, 4], dtype=np.float32),
                        np.array([5, 6], dtype=np.float32),
                    ],
                    "logits": [
                        np.array([[9.0, 1.2, 3.3], [0.7, 0.6, 5.6]], dtype=np.float32),
                        np.array([[0.9, 2.1, 3.3], [7.0, 6.0, 6.5]], dtype=np.float32),
                    ],
                },
                {
                    "loss": 2.0,
                    "tag_ids": np.array([3, 4, 5, 6], dtype=np.float32),
                    "logits": np.array(
                        [
                            [9.0, 1.2, 3.3],
                            [0.7, 0.6, 5.6],
                            [0.9, 2.1, 3.3],
                            [7.0, 6.0, 6.5],
                        ],
                        dtype=np.float32,
                    ),
                },
            ),
        ],
    )
    def test_convert_and_combine(
        self,
        outputs: List[Union[torch.Tensor, Dict[str, Any]]],
        np_batch: Dict[str, List[np.array]],
        np_epoch: Dict[str, Union[np.number, np.array]],
    ) -> None:
        # 1. output -> np_batch
        test_np_batch = self.ner_model_evaluation._convert_output_to_np_batch(outputs)
        for key in np_batch.keys():
            for _test, _true in zip(test_np_batch[key], np_batch[key]):
                assert np.array_equal(
                    _test, _true
                ), f"test_np_batch[{key}] = {_test} != {_true}"

        # 2. np_batch -> np_epoch
        test_np_epoch = self.ner_model_evaluation._combine_np_batch_to_np_epoch(
            np_batch
        )
        for key in np_epoch.keys():
            if key == "loss":
                assert (
                    test_np_epoch[key] == np_epoch[key]
                ), f"test_np_epoch[{key}] = {test_np_epoch[key]} != {np_epoch[key]}"
            else:
                for _test, _true in zip(test_np_epoch[key], np_epoch[key]):
                    assert np.array_equal(
                        _test, _true
                    ), f"test_np_epoch[{key}] = {_test} != {_true}"

    @pytest.mark.parametrize(
        "_np_tag_ids, " "_np_logits, " "true_flat, " "pred_flat",
        [
            (
                np.array([[0, 1], [1, 0]]),
                np.array([[[0.8, 0.2], [0.8, 0.2]], [[0.3, 0.7], [0.3, 0.7]]]),
                np.array([0, 1, 1, 0]),
                np.array([0, 0, 1, 1]),
            ),
            (
                np.array([0, 1, 1, 1]),
                np.array(
                    [
                        [9.0, 1.2, 3.3],
                        [0.7, 0.6, 5.6],
                        [0.9, 2.1, 3.3],
                        [7.0, 6.0, 6.5],
                    ]
                ),
                np.array([0, 1, 1, 1], dtype=int),
                np.array([0, 2, 2, 0], dtype=int),
            ),
        ],
    )
    def test_reduce_and_flatten(
        self,
        _np_tag_ids: np.array,
        _np_logits: np.array,
        true_flat: np.array,
        pred_flat: np.array,
    ) -> None:
        test_true_flat, test_pred_flat = self.ner_model_evaluation._reduce_and_flatten(
            _np_tag_ids, _np_logits
        )
        assert np.array_equal(
            test_true_flat, true_flat
        ), f"true_flat: test = {test_true_flat} != {true_flat} = true"
        assert np.array_equal(
            test_pred_flat, pred_flat
        ), f"pred_flat: test = {test_true_flat} != {pred_flat} = true"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_np_tag_ids, " "tags",
        [
            (
                np.array([0, 2, 1, 2]),
                np.array(["O", "PER", "ORG", "PER"]),
            ),
            (
                np.array([-100, 2, 1, -100]),
                np.array(["[S]", "PER", "ORG", "[S]"]),
            ),
        ],
    )
    def test_convert_tag_ids_to_tags(
        self,
        _np_tag_ids: np.array,
        tags: np.array,
    ) -> None:
        test_tags = self.ner_model_evaluation._convert_tag_ids_to_tags(_np_tag_ids)
        assert np.array_equal(test_tags, tags), f"test = {test_tags} != {tags} = true"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_tags, " "tags_new",
        [
            (
                {
                    "true": np.array(["[S]", "PER", "ORG", "[S]"]),
                    "pred": np.array(["PER", "ORG", "O", "O"]),
                },
                {
                    "true": np.array(["PER", "ORG"]),
                    "pred": np.array(["ORG", "O"]),
                },
            ),
        ],
    )
    def test_get_rid_of_special_tag_occurrences(
        self,
        _tags: Dict[str, np.array],
        tags_new: Dict[str, np.array],
    ) -> None:
        test_tags_new = self.ner_model_evaluation._get_rid_of_special_tag_occurrences(
            _tags
        )
        for k in tags_new.keys():
            assert np.array_equal(
                test_tags_new[k], tags_new[k]
            ), f"key = {k}: test = {test_tags_new[k]} != {tags_new[k]} = true"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "_tags_plain, " "_tag_subset, " "_filtered_tags, " "_filtered_tags_index",
        [
            (
                {"true": np.array(["O", "ORG", "PER"]), "pred": np.array([])},
                "all",
                ["O", "ORG", "PER"],
                None,
            ),
            (
                {"true": np.array(["O", "ORG", "PER"]), "pred": np.array([])},
                "fil",
                ["ORG", "PER"],
                None,
            ),
            (
                {"true": np.array(["O", "ORG", "PER"]), "pred": np.array([])},
                "O",
                ["O"],
                None,
            ),
            (
                {"true": np.array(["O", "ORG", "PER"]), "pred": np.array([])},
                "PER",
                ["PER"],
                1,
            ),
            (
                {"true": np.array(["O", "ORG", "PER"]), "pred": np.array([])},
                "ORG",
                ["ORG"],
                0,
            ),
            (
                {"true": np.array(["ORG"]), "pred": np.array([])},
                "all",
                ["O", "ORG", "PER"],
                None,
            ),
            (
                {"true": np.array(["ORG"]), "pred": np.array([])},
                "fil",
                ["ORG", "PER"],
                None,
            ),
            ({"true": np.array(["ORG"]), "pred": np.array([])}, "O", ["O"], None),
            ({"true": np.array(["ORG"]), "pred": np.array([])}, "PER", ["PER"], None),
            ({"true": np.array(["ORG"]), "pred": np.array([])}, "ORG", ["ORG"], 0),
            ({"true": np.array(["PER"]), "pred": np.array([])}, "PER", ["PER"], 0),
            ({"true": np.array(["PER"]), "pred": np.array([])}, "ORG", ["ORG"], None),
        ],
    )
    def test_get_filtered_tags(
        self,
        _tags_plain: Dict[str, np.array],
        _tag_subset: str,
        _filtered_tags: List[str],
        _filtered_tags_index: Optional[int],
    ) -> None:
        (
            test_filtered_tags,
            test_filtered_tags_index,
        ) = self.ner_model_evaluation._get_filtered_tags(_tag_subset, _tags_plain)

        assert (
            test_filtered_tags == _filtered_tags
        ), f"test_filtered_tags = {test_filtered_tags} != {_filtered_tags}"
        assert (
            test_filtered_tags_index == _filtered_tags_index
        ), f"test_filtered_tags_index = {test_filtered_tags_index} != {_filtered_tags_index}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "annotation_scheme, " "_tags, " "_phase, " "tag_subset, " "metrics",
        [
            (
                "plain",
                {
                    "true": np.array(["O", "PER", "ORG", "PER"]),
                    "pred": np.array(["O", "PER", "ORG", "PER"]),
                },
                "test",
                "fil",
                {
                    "entity_fil_numberofclasses_macro": 2,
                    "entity_fil_recall_micro": 1.0,
                    "entity_fil_recall_macro": 1.0,
                    "entity_fil_precision_micro": 1.0,
                    "entity_fil_precision_macro": 1.0,
                    "entity_fil_f1_micro": 1.0,
                    "entity_fil_f1_macro": 1.0,
                    "token_fil_numberofclasses_macro": 2,
                    "token_fil_recall_micro": 1.0,
                    "token_fil_recall_macro": 1.0,
                    "token_fil_precision_micro": 1.0,
                    "token_fil_precision_macro": 1.0,
                    "token_fil_f1_micro": 1.0,
                    "token_fil_f1_macro": 1.0,
                },
            ),
            (
                "plain",
                {
                    "true": np.array(["O", "PER", "ORG", "PER"]),
                    "pred": np.array(["O", "PER", "ORG", "PER"]),
                },
                "test",
                "PER",
                {
                    "entity_PER_recall": 1.0,
                    "entity_PER_precision": 1.0,
                    "entity_PER_f1": 1.0,
                    "token_PER_recall": 1.0,
                    "token_PER_precision": 1.0,
                    "token_PER_f1": 1.0,
                },
            ),
            (
                "bio",
                {
                    "true": np.array(["O", "B-PER", "I-PER", "O", "B-PER"]),
                    "pred": np.array(["O", "B-PER", "I-PER", "I-PER", "B-PER"]),
                },
                "test",
                "PER",
                {
                    "entity_PER_recall": 0.5,
                    "entity_PER_precision": 0.5,
                    "entity_PER_f1": 0.5,
                    "token_PER_recall": 1.0,
                    "token_PER_precision": 0.75,
                    "token_PER_f1": 0.86,
                },
            ),
            (
                "bio",
                {
                    "true": np.array(["O", "B-PER", "B-ORG", "O"]),
                    "pred": np.array(["O", "B-ORG", "I-ORG", "O"]),
                },
                "test",
                "PER",
                {
                    "entity_PER_recall": 0.0,
                    "entity_PER_precision": -1.0,
                    "entity_PER_f1": -1.0,
                    "token_PER_recall": 0.0,
                    "token_PER_precision": -1.0,
                    "token_PER_f1": -1.0,
                },
            ),
            (
                "bio",
                {
                    "true": np.array(["O", "B-PER", "B-ORG", "O"]),
                    "pred": np.array(["O", "B-ORG", "I-ORG", "O"]),
                },
                "test",
                "ORG",
                {
                    "entity_ORG_recall": 0.0,
                    "entity_ORG_precision": 0.0,
                    "entity_ORG_f1": 0.0,
                    "token_ORG_recall": 1.0,
                    "token_ORG_precision": 0.5,
                    "token_ORG_f1": 0.66,
                },
            ),
            (
                "bio",
                {
                    "true": np.array(["O", "B-PER", "B-ORG", "O"]),
                    "pred": np.array(["O", "B-ORG", "B-ORG", "O"]),
                },
                "test",
                "PER",
                {
                    "entity_PER_recall": 0.0,
                    "entity_PER_precision": -1.0,
                    "entity_PER_f1": -1.0,
                    "token_PER_recall": 0.0,
                    "token_PER_precision": -1.0,
                    "token_PER_f1": -1.0,
                },
            ),
        ],
    )
    def test_compute_metrics_for_tags_subset(
        self,
        annotation_scheme: str,
        _tags: Dict[str, np.array],
        _phase: str,
        tag_subset: str,
        metrics: Dict[str, float],
    ) -> None:
        if annotation_scheme == "plain":
            test_metrics = self.ner_model_evaluation._compute_metrics_for_tags_subset(
                _tags, _phase, tag_subset
            )
        elif annotation_scheme == "bio":
            test_metrics = (
                self.ner_model_evaluation_bio._compute_metrics_for_tags_subset(
                    _tags, _phase, tag_subset
                )
            )
        else:
            raise Exception(f"annotation_scheme = {annotation_scheme} unknown.")

        a = set(list(test_metrics.keys()))
        b = set(list(metrics.keys()))
        assert len(a.difference(b)) == 0, a.difference(b)
        assert len(b.difference(a)) == 0, b.difference(a)

        for k in list(set(list(test_metrics.keys()) + list(metrics.keys()))):
            assert test_metrics[k] == pytest_approx(
                metrics[k]
            ), f"test_metrics[{k}] = {test_metrics[k]} != {metrics[k]}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "annotation_scheme, " "phase, " "np_epoch, " "epoch_metrics, " "epoch_tags",
        [
            (
                "plain",
                "test",
                {
                    "loss": 2.0,
                    "tag_ids": np.array(
                        [0, 2, 1, 0], dtype=np.float32
                    ),  # [O, PER, ORG, O]  (true)
                    "logits": np.array(
                        [  # O, ORG, PER
                            [9.0, 3.3, 1.2],  # O
                            [0.7, 5.6, 0.6],  # ORG
                            [0.9, 3.3, 2.1],  # ORG
                            [7.0, 6.5, 6.0],  # O
                        ],
                        dtype=np.float32,
                    ),
                },
                {
                    "entity_fil_numberofclasses_macro": 1,
                    "entity_fil_precision_micro": 0.0,
                    "entity_fil_precision_macro": 0.0,
                    "entity_fil_recall_micro": 0.0,
                    "entity_fil_recall_macro": 0.0,
                    "entity_fil_f1_micro": 0.0,
                    "entity_fil_f1_macro": 0.0,
                    "entity_PER_precision": -1.0,
                    "entity_PER_recall": 0.0,
                    "entity_PER_f1": -1.0,
                    "entity_ORG_precision": 0.0,
                    "entity_ORG_recall": 0.0,
                    "entity_ORG_f1": 0.0,
                    "token_all_loss": 2.0,
                    "token_all_acc": 0.75,
                    "token_all_precision_micro": 0.75,
                    "token_all_precision_macro": -1.0,
                    "token_all_recall_micro": 0.75,
                    "token_all_recall_macro": 0.66,
                    "token_all_f1_micro": 0.75,
                    "token_all_f1_macro": -1.0,
                    "token_fil_numberofclasses_macro": 1,
                    "token_fil_precision_micro": 0.5,
                    "token_fil_precision_macro": -1.0,
                    "token_fil_recall_micro": 0.5,
                    "token_fil_recall_macro": 0.5,
                    "token_fil_f1_micro": 0.5,
                    "token_fil_f1_macro": -1.0,
                    "token_O_precision": 1.0,
                    "token_O_recall": 1.0,
                    "token_O_f1": 1.0,
                    "token_PER_precision": -1.0,
                    "token_PER_recall": 0.0,
                    "token_PER_f1": -1.0,
                    "token_ORG_precision": 0.5,
                    "token_ORG_recall": 1.0,
                    "token_ORG_f1": 0.66,
                },
                {
                    "true": np.array(["O", "PER", "ORG", "O"]),
                    "pred": np.array(["O", "ORG", "ORG", "O"]),
                },
            ),
            (
                "bio",
                "test",
                {
                    "loss": 2.0,
                    "tag_ids": np.array(
                        [0, 2, 1, 0], dtype=np.float32
                    ),  # [O, B-PER, B-ORG, O]  (true)
                    "logits": np.array(
                        [
                            # 0, B-ORG, B-PER, I-ORG, I-PER  (pred)
                            [9.0, 3.3, 1.2, 1.0, 1.0],  # O
                            [0.7, 5.6, 1.2, 1.0, 1.0],  # B-ORG
                            [0.9, 1.0, 2.1, 3.3, 1.0],  # I-ORG
                            [7.0, 6.5, 6.9, 1.0, 1.0],  # O
                        ],
                        dtype=np.float32,
                    ),
                },
                {
                    "entity_fil_numberofclasses_macro": 1,
                    "entity_fil_precision_micro": 0.0,
                    "entity_fil_precision_macro": 0.0,
                    "entity_fil_recall_micro": 0.0,
                    "entity_fil_recall_macro": 0.0,
                    "entity_fil_f1_micro": 0.0,
                    "entity_fil_f1_macro": 0.0,
                    "entity_PER_precision": -1.0,
                    "entity_PER_recall": 0.0,
                    "entity_PER_f1": -1.0,
                    "entity_ORG_precision": 0.0,
                    "entity_ORG_recall": 0.0,
                    "entity_ORG_f1": 0.0,
                    "token_all_loss": 2.0,
                    "token_all_acc": 0.75,
                    "token_all_precision_micro": 0.75,
                    "token_all_precision_macro": -1.0,
                    "token_all_recall_micro": 0.75,
                    "token_all_recall_macro": 0.66,
                    "token_all_f1_micro": 0.75,
                    "token_all_f1_macro": -1.0,
                    "token_fil_numberofclasses_macro": 1,
                    "token_fil_precision_micro": 0.5,
                    "token_fil_precision_macro": -1.0,
                    "token_fil_recall_micro": 0.5,
                    "token_fil_recall_macro": 0.5,
                    "token_fil_f1_micro": 0.5,
                    "token_fil_f1_macro": -1.0,
                    "token_O_precision": 1.0,
                    "token_O_recall": 1.0,
                    "token_O_f1": 1.0,
                    "token_PER_precision": -1.0,
                    "token_PER_recall": 0.0,
                    "token_PER_f1": -1.0,
                    "token_ORG_precision": 0.5,
                    "token_ORG_recall": 1.0,
                    "token_ORG_f1": 0.66,
                },
                {
                    "true": np.array(["O", "B-PER", "B-ORG", "O"]),
                    "pred": np.array(["O", "B-ORG", "I-ORG", "O"]),
                },
            ),
        ],
    )
    def test_compute_metrics(
        self,
        annotation_scheme: str,
        phase: str,
        np_epoch: Dict[str, Union[np.number, np.array]],
        epoch_metrics: Dict[str, np.array],
        epoch_tags: Dict[str, np.array],
    ) -> None:
        if annotation_scheme == "plain":
            (
                test_epoch_metrics,
                test_epoch_tags,
            ) = self.ner_model_evaluation._compute_metrics(phase, np_epoch)
        elif annotation_scheme == "bio":
            (
                test_epoch_metrics,
                test_epoch_tags,
            ) = self.ner_model_evaluation_bio._compute_metrics(phase, np_epoch)
        else:
            raise Exception(f"annotation_scheme = {annotation_scheme} unknown.")

        a = set(list(test_epoch_metrics.keys()))
        b = set(list(epoch_metrics.keys()))
        assert len(a.difference(b)) == 0, a.difference(b)
        assert len(b.difference(a)) == 0, b.difference(a)

        for k in list(set(list(test_epoch_tags.keys()) + list(epoch_tags.keys()))):
            assert np.array_equal(
                test_epoch_tags[k], epoch_tags[k]
            ), f"key = {k}: test_epoch_tags = {test_epoch_tags[k]} != {epoch_tags[k]}"

        for k in list(
            set(list(test_epoch_metrics.keys()) + list(epoch_metrics.keys()))
        ):
            assert test_epoch_metrics[k] == pytest_approx(
                epoch_metrics[k]
            ), f"key = {k}: test_epoch_metrics = {test_epoch_metrics[k]} != {epoch_metrics[k]}"

    ####################################################################################################################
    @pytest.mark.parametrize(
        "phase, " "outputs, " "epoch_metrics, " "epoch_loss",
        [
            (
                "test",
                [
                    [
                        torch.tensor([1.0, 2.0]),
                        torch.tensor([0, 2]),  # =          [O, PER]  (true)
                        torch.tensor(
                            [[9.0, 1.2, 3.3], [0.7, 5.6, 0.6]]
                        ),  # = [0, 1] = [O, ORG]  (pred)
                    ],
                    [
                        torch.tensor([2.0, 3.0]),
                        torch.tensor([1, 0]),  # =          [ORG, O]  (true)
                        torch.tensor(
                            [[0.9, 3.3, 2.1], [7.0, 6.5, 6.0]]
                        ),  # = [1, 0] = [ORG, O]  (pred)
                    ],
                ],
                {
                    "entity_fil_numberofclasses_macro": 1,
                    "entity_fil_precision_micro": 0.0,
                    "entity_fil_precision_macro": 0.0,
                    "entity_fil_recall_micro": 0.0,
                    "entity_fil_recall_macro": 0.0,
                    "entity_fil_f1_micro": 0.0,
                    "entity_fil_f1_macro": 0.0,
                    "entity_PER_precision": -1.0,
                    "entity_PER_recall": 0.0,
                    "entity_PER_f1": -1.0,
                    "entity_ORG_precision": 0.0,
                    "entity_ORG_recall": 0.0,
                    "entity_ORG_f1": 0.0,
                    "token_all_loss": 2.0,
                    "token_all_acc": 0.75,
                    "token_all_precision_micro": 0.75,
                    "token_all_precision_macro": -1.0,
                    "token_all_recall_micro": 0.75,
                    "token_all_recall_macro": 0.66,
                    "token_all_f1_micro": 0.75,
                    "token_all_f1_macro": -1.0,
                    "token_fil_numberofclasses_macro": 1,
                    "token_fil_precision_micro": 0.5,
                    "token_fil_precision_macro": -1.0,
                    "token_fil_recall_micro": 0.5,
                    "token_fil_recall_macro": 0.5,
                    "token_fil_f1_micro": 0.5,
                    "token_fil_f1_macro": -1.0,
                    "token_O_precision": 1.0,
                    "token_O_recall": 1.0,
                    "token_O_f1": 1.0,
                    "token_PER_precision": -1.0,
                    "token_PER_recall": 0.0,
                    "token_PER_f1": -1.0,
                    "token_ORG_precision": 0.5,
                    "token_ORG_recall": 1.0,
                    "token_ORG_f1": 0.66,
                },
                2.0,
            ),
        ],
    )
    def test_execute(
        self,
        phase: str,
        outputs: List[Union[torch.Tensor, Dict[str, Any]]],
        epoch_metrics: Dict[str, np.array],
        epoch_loss: float,
    ) -> None:
        test_epoch_metrics, _, _, test_epoch_loss = self.ner_model_evaluation.execute(
            phase, outputs
        )

        a = set(list(test_epoch_metrics.keys()))
        b = set(list(epoch_metrics.keys()))
        assert len(a.difference(b)) == 0, a.difference(b)
        assert len(b.difference(a)) == 0, b.difference(a)

        for k in list(
            set(list(test_epoch_metrics.keys()) + list(epoch_metrics.keys()))
        ):
            assert test_epoch_metrics[k] == pytest_approx(
                epoch_metrics[k]
            ), f"key = {k}: test_epoch_metrics = {test_epoch_metrics[k]} != {epoch_metrics[k]}"


if __name__ == "__main__":
    test = TestNerModelEvaluation()
    test.test_convert_and_combine()
    test.test_reduce_and_flatten()
    test.test_convert_tag_ids_to_tags()
    test.test_get_rid_of_special_tag_occurrences()
    test.test_get_filtered_tags()
    test.test_compute_metrics_for_tags_subset()
    test.test_compute_metrics()
    test.test_execute()
