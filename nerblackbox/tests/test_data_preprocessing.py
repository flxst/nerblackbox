
import pytest
import torch
from transformers import AutoTokenizer
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example_to_tensors import (
    InputExampleToTensors,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.bert_dataset import (
    BertDataset,
)

tokenizer = AutoTokenizer.from_pretrained(
    "af-ai-center/bert-base-swedish-uncased",
    do_lower_case=False,
    additional_special_tokens=["[newline]", "[NEWLINE]"],
    use_fast=True,
)


class TestDataPreprocessing:

    ####################################################################################################################
    @pytest.mark.parametrize(
        "text, "
        "labels, "
        "max_seq_length, "
        "true_input_ids, "
        "true_attention_mask, "
        "true_segment_ids, "
        "true_tag_ids, "
        "true_input_tokens",
        [
            (
                "arbetsförmedlingen ai-center finns i stockholm",
                "ORG ORG O O LOC",
                12,
                torch.tensor([101, 7093, 2842, 8126, 1011, 5410, 1121, 1045, 1305, 102, 0, 0]),
                torch.tensor([1]*10 + [0]*2),
                torch.tensor([0]*12),
                torch.tensor([-100, 1, -100, 1, -100, -100, 0, 0, 2, -100, -100, -100]),
                ["[CLS]", "arbetsförmedl", "##ingen", "ai", "-", "center", "finns", "i", "stockholm", "[SEP]"]
                + ["[PAD]"]*2,
            ),
            (
                    "arbetsförmedlingen ai-center finns i stockholm",
                    "ORG ORG O O LOC",
                    4,
                    torch.tensor([101, 7093, 2842, 102]),
                    torch.tensor([1] * 4),
                    torch.tensor([0] * 4),
                    torch.tensor([-100, 1, -100, -100]),
                    ["[CLS]", "arbetsförmedl", "##ingen", "[SEP]"]
            ),
        ]
    )
    def tests(
            self,
            text: str,
            labels: str,
            max_seq_length: int,
            true_input_ids: torch.tensor,
            true_attention_mask: torch.tensor,
            true_segment_ids: torch.tensor,
            true_tag_ids: torch.tensor,
            true_input_tokens: torch.tensor,
    ) -> None:

        ##################################
        # 1. InputExampleToTensors
        ##################################
        input_example = InputExample(
            guid="",
            text=text,
            tags=labels,
        )
        input_example_to_tensors = InputExampleToTensors(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            tag_tuple=("O", "ORG", "LOC"),
        )
        input_ids, attention_mask, segment_ids, tag_ids = input_example_to_tensors(input_example)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for (_test, true, string) in zip(
                [input_ids, attention_mask, segment_ids, tag_ids, input_tokens],
                [true_input_ids, true_attention_mask, true_segment_ids, true_tag_ids, true_input_tokens],
                ["input_ids", "attention_mask", "segment_ids", "tag_ids", "input_tokens"],
        ):
            assert list(_test) == list(true), f"{string} = {_test} != {true}"

        ##################################
        # 2. BertDataset
        ##################################
        data = BertDataset(
            input_examples=[input_example], transform=input_example_to_tensors
        )
        assert len(data) == 1, f"len(data) = {len(data)} != 1"
        for i, (_test, true, string) in enumerate(zip(
                data[0],
                [true_input_ids, true_attention_mask, true_segment_ids, true_tag_ids],
                ["input_ids", "attention_mask", "segment_ids", "tag_ids"],
        )):
            assert list(_test) == list(true), f"{string} = {_test} != {true}"


if __name__ == "__main__":
    test = TestDataPreprocessing()
    test.tests()
