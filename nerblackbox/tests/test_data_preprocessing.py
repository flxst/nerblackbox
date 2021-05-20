
import pytest
import torch
from transformers import AutoTokenizer
from typing import List, Dict
from pkg_resources import resource_filename
from nerblackbox.modules.ner_training.data_preprocessing.tools.csv_reader import (
    CsvReader,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_example import (
    InputExample,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.input_examples_to_tensors import (
    InputExamplesToTensors,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.bert_dataset import (
    BertDataset,
)
from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import EncodingsKeys
from nerblackbox.tests.utils import PseudoDefaultLogger

tokenizer = AutoTokenizer.from_pretrained(
    "af-ai-center/bert-base-swedish-uncased",
    do_lower_case=False,
    additional_special_tokens=["[newline]", "[NEWLINE]"],
    use_fast=True,
)


class TestCsvReaderAndDataProcessor:

    csv_reader = CsvReader(
        path=resource_filename("nerblackbox", "tests/test_data"),
        tokenizer=tokenizer,
        do_lower_case=False,
        csv_file_separator="\t",
        default_logger=None,
    )

    ####################################################################################################################
    @pytest.mark.parametrize(
        "tag_list, "
        "data",
        [
            (
                    ["O", "PER", "ORG", "MISC"],
                    {
                        "train": [InputExample(
                            guid="",
                            text="På skidspår.se kan längdskidåkare själva betygsätta förhållandena i spåren .",
                            tags="O MISC O O O O O O O O")],
                        "val": [InputExample(
                            guid="",
                            text="Fastigheten är ett landmärke designad av arkitekten Robert Stern .",
                            tags="O O O O O O O PER PER O")],
                        "test": [InputExample(
                            guid="",
                            text="Apple noteras för 85,62 poäng , vilket är den högsta siffran någonsin i undersökningen .",
                            tags="ORG O O O O O O O O O O O O O O")],
                    }
            ),
        ]
    )
    def tests(
            self,
            tag_list: List[str],
            data: Dict[str, List[InputExample]],
    ) -> None:

        ##################################
        # 1. CsvReader
        ##################################
        for phase in ["train", "val", "test"]:
            test_data = self.csv_reader.get_input_examples(phase)
            assert len(test_data) == 1, f"ERROR! len(test_data) = {len(test_data)} should be 1."
            assert test_data[0].text == data[phase][0].text, \
                f"phase = {phase}: test_data_text = {test_data[0].text} != {data[phase][0].text}"
            assert test_data[0].tags == data[phase][0].tags, \
                f"phase = {phase}: test_data_tags = {test_data[0].tags} != {data[phase][0].tags}"

        ##################################
        # 2. DataProcessor
        ##################################
        data_preprocessor = DataPreprocessor(
            tokenizer=tokenizer,
            do_lower_case=False,
            default_logger=PseudoDefaultLogger(),
            max_seq_length=4,
        )
        test_data, test_tag_list = data_preprocessor.get_input_examples_train(dataset_name=None, prune_ratio=None)
        assert set(test_tag_list) == set(tag_list), f"test_tag_list = {test_tag_list} != {tag_list}"
        for phase in ["train", "val", "test"]:
            assert len(test_data[phase]) == 1, f"ERROR! len(test_data[{phase}]) = {len(test_data[phase])} should be 1."
            assert test_data[phase][0].text == data[phase][0].text, \
                f"phase = {phase}: test_data.text = {test_data[phase].text} != {data[phase][0].text}"
            assert test_data[phase][0].tags == data[phase][0].tags, \
                f"phase = {phase}: test_data.tags = {test_data[phase].tags} != {data[phase][0].tags}"


class TestInputExamplesToTensorsAndBertDataset:

    @pytest.mark.parametrize(
        "texts, "
        "labels, "
        "tag_tuple, "
        "max_seq_length, "
        "true_input_ids, "
        "true_attention_mask, "
        "true_token_type_ids, "
        "true_tag_ids, "
        "true_input_tokens",
        [
            # 1. single example: no truncation
            (
                ["arbetsförmedlingen ai-center finns i stockholm"],
                ["ORG ORG O O LOC"],
                ("O", "ORG", "LOC"),
                12,
                torch.tensor([[101, 7093, 2842, 8126, 1011, 5410, 1121, 1045, 1305, 102, 0, 0]]),
                torch.tensor([[1]*10 + [0]*2]),
                torch.tensor([[0]*12]),
                torch.tensor([[-100, 1, -100, 1, -100, -100, 0, 0, 2, -100, -100, -100]]),
                [
                    ["[CLS]", "arbetsförmedl", "##ingen", "ai", "-", "center", "finns", "i", "stockholm", "[SEP]"]
                    + ["[PAD]"]*2
                ],
            ),
            # 2. single example: truncation
            (
                    ["arbetsförmedlingen ai-center finns i stockholm"],
                    ["ORG ORG O O LOC"],
                    ("O", "ORG", "LOC"),
                    4,
                    torch.tensor([
                        [101, 7093, 2842, 102],
                        [101, 8126, 1011, 102],
                        [101, 5410, 1121, 102],
                        [101, 1045, 1305, 102],
                    ]),
                    torch.tensor([
                        [1] * 4,
                        [1] * 4,
                        [1] * 4,
                        [1] * 4,
                    ]),
                    torch.tensor([
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                    ]),
                    torch.tensor([
                        [-100, 1, -100, -100],
                        [-100, 1, -100, -100],
                        [-100, -100, 0, -100],
                        [-100, 0,    2, -100],
                    ]),
                    [
                        ["[CLS]", "arbetsförmedl",   "##ingen", "[SEP]"],
                        ["[CLS]",            "ai",         "-", "[SEP]"],
                        ["[CLS]",        "center",     "finns", "[SEP]"],
                        ["[CLS]",             "i", "stockholm", "[SEP]"],
                    ]
            ),
            # 3. two examples: truncation
            (
                    ["arbetsförmedlingen ai-center", "finns i stockholm"],
                    ["ORG ORG", "O O LOC"],
                    ("O", "ORG", "LOC"),
                    4,
                    torch.tensor([
                        [101, 7093, 2842, 102],
                        [101, 8126, 1011, 102],
                        [101, 5410,  102,   0],
                        [101, 1121, 1045, 102],
                        [101, 1305,  102,   0],
                    ]),
                    torch.tensor([
                        [1] * 4,
                        [1] * 4,
                        [1] * 3 + [0],
                        [1] * 4,
                        [1] * 3 + [0],
                        ]),
                    torch.tensor([
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        ]),
                    torch.tensor([
                        [-100,    1, -100, -100],
                        [-100,    1, -100, -100],
                        [-100, -100, -100, -100],
                        [-100,    0,    0, -100],
                        [-100,    2, -100, -100],
                    ]),
                    [
                        ["[CLS]", "arbetsförmedl", "##ingen", "[SEP]"],
                        ["[CLS]", "ai", "-", "[SEP]"],
                        ["[CLS]", "center", "[SEP]", "[PAD]"],
                        ["[CLS]", "finns", "i", "[SEP]"],
                        ["[CLS]", "stockholm", "[SEP]", "[PAD]"],
                    ]
            ),
        ]
    )
    def tests(
            self,
            texts: List[str],
            labels: List[str],
            tag_tuple: List[str],
            max_seq_length: int,
            true_input_ids: torch.tensor,
            true_attention_mask: torch.tensor,
            true_token_type_ids: torch.tensor,
            true_tag_ids: torch.tensor,
            true_input_tokens: torch.tensor,
    ) -> None:

        ##################################
        # 1. InputExamplesToTensors
        ##################################
        input_examples = [InputExample(
                guid="",
                text=text,
                tags=label,
            )
            for text, label in zip(texts, labels)
        ]
        input_examples_to_tensors = InputExamplesToTensors(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            tag_tuple=tuple(tag_tuple),
        )
        encodings = input_examples_to_tensors(input_examples)
        input_tokens = [tokenizer.convert_ids_to_tokens(input_ids_single) for input_ids_single in encodings["input_ids"]]

        for (string, true) in zip(
                EncodingsKeys,
                [true_input_ids, true_attention_mask, true_token_type_ids, true_tag_ids],
        ):
            assert torch.all(torch.eq(encodings[string], true)), f"{string} = {encodings[string]} != {true}"

        for (string, true, _test) in zip(
                ["input_tokens"],
                [true_input_tokens],
                [input_tokens],
        ):
            assert _test == true, f"{string} = {_test} != {true}"

        ##################################
        # 2. BertDataset
        ##################################
        data = BertDataset(encodings=encodings)  # data[j] = 4 torch tensors corresponding to EncodingKeys
        assert len(data) >= len(texts), f"len(data) = {len(data)} < {len(texts)} = len(texts)"
        for i, (string, true) in enumerate(zip(
                ["input_ids", "attention_mask", "token_type_ids", "tag_ids"],
                [true_input_ids, true_attention_mask, true_token_type_ids, true_tag_ids],
        )):
            for j in range(len(true)):
                assert torch.all(torch.eq(data[j][i], true[j])), \
                    f"{string} = {data[j][i]} != {true[j]}"


if __name__ == "__main__":
    for test in [TestCsvReaderAndDataProcessor(), TestInputExamplesToTensorsAndBertDataset()]:
        test.tests()
