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
    order_tag_list,
    convert_tag_list_bio2plain,
)
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    EncodingsKeys,
)
from nerblackbox.tests.utils import PseudoDefaultLogger
from nerblackbox.modules.ner_training.ner_model import NEWLINE_TOKENS

tokenizer = AutoTokenizer.from_pretrained(
    "af-ai-center/bert-base-swedish-uncased",
    do_lower_case=False,
    additional_special_tokens=NEWLINE_TOKENS,
    use_fast=True,
)

csv_reader = CsvReader(
    path=resource_filename("nerblackbox", "tests/test_data"),
    tokenizer=tokenizer,
    do_lower_case=False,
    csv_file_separator="\t",
    default_logger=None,
)

data_preprocessor = DataPreprocessor(
    tokenizer=tokenizer,
    do_lower_case=False,
    default_logger=PseudoDefaultLogger(),
    max_seq_length=4,
)


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestCsvReaderAndDataProcessor:

    ####################################################################################################################
    @pytest.mark.parametrize(
        "tag_list, " "input_examples",
        [
            (
                ["O", "PER", "ORG", "MISC"],
                {
                    "train": [
                        InputExample(
                            guid="",
                            text="På skidspår.se kan längdskidåkare själva betygsätta förhållandena i spåren .",
                            tags="O MISC O O O O O O O O",
                        ),
                    ],
                    "val": [
                        InputExample(
                            guid="",
                            text="Fastigheten är ett landmärke designad av arkitekten Robert Stern .",
                            tags="O O O O O O O PER PER O",
                        ),
                    ],
                    "test": [
                        InputExample(
                            guid="",
                            text="Apple noteras för 85,62 poäng , vilket är den högsta siffran någonsin i undersökningen .",
                            tags="ORG O O O O O O O O O O O O O O",
                        ),
                    ],
                    "predict": [
                        InputExample(
                            guid="",
                            text="På skidspår.se kan längdskidåkare själva betygsätta förhållandena i spåren .",
                            tags="O MISC O O O O O O O O",
                        ),
                        InputExample(
                            guid="",
                            text="Fastigheten är ett landmärke designad av arkitekten Robert Stern .",
                            tags="O O O O O O O PER PER O",
                        ),
                        InputExample(
                            guid="",
                            text="Apple noteras för 85,62 poäng , vilket är den högsta siffran någonsin i undersökningen .",
                            tags="ORG O O O O O O O O O O O O O O",
                        ),
                    ],
                },
            ),
        ],
    )
    def tests(
        self,
        tag_list: List[str],
        input_examples: Dict[str, List[InputExample]],
    ) -> None:

        ##################################
        # 1. CsvReader
        ##################################
        for phase in ["train", "val", "test"]:
            test_input_examples = csv_reader.get_input_examples(phase)
            assert (
                len(test_input_examples) == 1 or 2
            ), f"ERROR! len(test_input_examples) = {len(test_input_examples)} should be 1 or 2."
            assert (
                test_input_examples[0].text == input_examples[phase][0].text
            ), f"phase = {phase}: test_input_examples_text = {test_input_examples[0].text} != {input_examples[phase][0].text}"
            assert (
                test_input_examples[0].tags == input_examples[phase][0].tags
            ), f"phase = {phase}: test_input_examples_tags = {test_input_examples[0].tags} != {input_examples[phase][0].tags}"

        ##################################
        # 2. DataProcessor
        ##################################
        # a. get_input_examples_train
        test_input_examples, test_tag_list = data_preprocessor.get_input_examples_train(
            prune_ratio={"train": 0.5, "val": 1.0, "test": 1.0},
            dataset_name=None,
        )
        assert set(test_tag_list) == set(
            tag_list
        ), f"test_tag_list = {test_tag_list} != {tag_list}"
        for phase in ["train", "val", "test"]:
            assert (
                len(test_input_examples[phase]) == 1
            ), f"ERROR! len(test_input_examples[{phase}]) = {len(test_input_examples[phase])} should be 1."
            assert (
                test_input_examples[phase][0].text == input_examples[phase][0].text
            ), f"phase = {phase}: test_input_examples.text = {test_input_examples[phase].text} != {input_examples[phase][0].text}"
            assert (
                test_input_examples[phase][0].tags == input_examples[phase][0].tags
            ), f"phase = {phase}: test_input_examples.tags = {test_input_examples[phase].tags} != {input_examples[phase][0].tags}"

        # b. get_input_examples_predict
        test_sentences = [
            elem.text for v in test_input_examples.values() for elem in v
        ]  # retrieve example sentences
        test_input_examples_predict = data_preprocessor.get_input_examples_predict(
            test_sentences
        )["predict"]
        assert len(test_input_examples_predict) == len(
            input_examples["predict"]
        ), f"len(test_input_examples_predict) = {len(test_input_examples_predict)} != {len(input_examples['predict'])}"
        for (test_input_example_predict, true_input_example_predict) in zip(
            test_input_examples_predict, input_examples["predict"]
        ):
            assert (
                test_input_example_predict.text == true_input_example_predict.text
            ), f"test_input_example_predict.text = {test_input_example_predict.text} != {true_input_example_predict.text}"
            true_input_example_predict_tags = " ".join(
                "O" for _ in range(len(true_input_example_predict.text.split()))
            )
            assert (
                test_input_example_predict.tags == true_input_example_predict_tags
            ), f"test_input_example_predict.tags = {test_input_example_predict.tags} != {true_input_example_predict_tags}"

        # c. to_dataloader
        dataloader = data_preprocessor.to_dataloader(
            input_examples, tag_list, batch_size=1
        )
        for key in ["train", "val", "test", "predict"]:
            assert (
                key in dataloader.keys()
            ), f"key = {key} not in dataloader.keys() = {dataloader.keys()}"
            # TODO: further testing


########################################################################################################################
########################################################################################################################
########################################################################################################################
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
                torch.tensor(
                    [[101, 7093, 2842, 8126, 1011, 5410, 1121, 1045, 1305, 102, 0, 0]]
                ),
                torch.tensor([[1] * 10 + [0] * 2]),
                torch.tensor([[0] * 12]),
                torch.tensor(
                    [[-100, 1, -100, 1, -100, -100, 0, 0, 2, -100, -100, -100]]
                ),
                [
                    [
                        "[CLS]",
                        "arbetsförmedl",
                        "##ingen",
                        "ai",
                        "-",
                        "center",
                        "finns",
                        "i",
                        "stockholm",
                        "[SEP]",
                    ]
                    + ["[PAD]"] * 2
                ],
            ),
            # 2. single example: no truncation, [NEWLINE]
            (
                ["arbetsförmedlingen ai-center [NEWLINE] finns i stockholm"],
                ["ORG ORG O O O LOC"],
                ("O", "ORG", "LOC"),
                12,
                torch.tensor(
                    [
                        [
                            101,
                            7093,
                            2842,
                            8126,
                            1011,
                            5410,
                            30523,
                            1121,
                            1045,
                            1305,
                            102,
                            0,
                        ]
                    ]
                ),
                torch.tensor([[1] * 11 + [0] * 1]),
                torch.tensor([[0] * 12]),
                torch.tensor([[-100, 1, -100, 1, -100, -100, 0, 0, 0, 2, -100, -100]]),
                [
                    [
                        "[CLS]",
                        "arbetsförmedl",
                        "##ingen",
                        "ai",
                        "-",
                        "center",
                        "[NEWLINE]",
                        "finns",
                        "i",
                        "stockholm",
                        "[SEP]",
                    ]
                    + ["[PAD]"] * 1
                ],
            ),
            # 3. single example: truncation
            (
                ["arbetsförmedlingen ai-center finns i stockholm"],
                ["ORG ORG O O LOC"],
                ("O", "ORG", "LOC"),
                4,
                torch.tensor(
                    [
                        [101, 7093, 2842, 102],
                        [101, 8126, 1011, 102],
                        [101, 5410, 1121, 102],
                        [101, 1045, 1305, 102],
                    ]
                ),
                torch.tensor(
                    [
                        [1] * 4,
                        [1] * 4,
                        [1] * 4,
                        [1] * 4,
                    ]
                ),
                torch.tensor(
                    [
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                    ]
                ),
                torch.tensor(
                    [
                        [-100, 1, -100, -100],
                        [-100, 1, -100, -100],
                        [-100, -100, 0, -100],
                        [-100, 0, 2, -100],
                    ]
                ),
                [
                    ["[CLS]", "arbetsförmedl", "##ingen", "[SEP]"],
                    ["[CLS]", "ai", "-", "[SEP]"],
                    ["[CLS]", "center", "finns", "[SEP]"],
                    ["[CLS]", "i", "stockholm", "[SEP]"],
                ],
            ),
            # 4. two examples: truncation
            (
                ["arbetsförmedlingen ai-center", "finns i stockholm"],
                ["ORG ORG", "O O LOC"],
                ("O", "ORG", "LOC"),
                4,
                torch.tensor(
                    [
                        [101, 7093, 2842, 102],
                        [101, 8126, 1011, 102],
                        [101, 5410, 102, 0],
                        [101, 1121, 1045, 102],
                        [101, 1305, 102, 0],
                    ]
                ),
                torch.tensor(
                    [
                        [1] * 4,
                        [1] * 4,
                        [1] * 3 + [0],
                        [1] * 4,
                        [1] * 3 + [0],
                    ]
                ),
                torch.tensor(
                    [
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                        [0] * 4,
                    ]
                ),
                torch.tensor(
                    [
                        [-100, 1, -100, -100],
                        [-100, 1, -100, -100],
                        [-100, -100, -100, -100],
                        [-100, 0, 0, -100],
                        [-100, 2, -100, -100],
                    ]
                ),
                [
                    ["[CLS]", "arbetsförmedl", "##ingen", "[SEP]"],
                    ["[CLS]", "ai", "-", "[SEP]"],
                    ["[CLS]", "center", "[SEP]", "[PAD]"],
                    ["[CLS]", "finns", "i", "[SEP]"],
                    ["[CLS]", "stockholm", "[SEP]", "[PAD]"],
                ],
            ),
        ],
    )
    def tests(
        self,
        texts: List[str],
        labels: List[str],
        tag_tuple: List[str],
        max_seq_length: int,
        true_input_ids: torch.Tensor,
        true_attention_mask: torch.Tensor,
        true_token_type_ids: torch.Tensor,
        true_tag_ids: torch.Tensor,
        true_input_tokens: torch.Tensor,
    ) -> None:

        ##################################
        # 1. InputExamplesToTensors
        ##################################
        input_examples = [
            InputExample(
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
            default_logger=PseudoDefaultLogger(),
        )
        encodings = input_examples_to_tensors(input_examples, predict=False)
        input_tokens = [
            tokenizer.convert_ids_to_tokens(input_ids_single)
            for input_ids_single in encodings["input_ids"]
        ]

        for (string, true) in zip(
            EncodingsKeys,
            [true_input_ids, true_attention_mask, true_token_type_ids, true_tag_ids],
        ):
            assert torch.all(
                torch.eq(encodings[string], true)
            ), f"{string} = {encodings[string]} != {true}"

        for (string, true, _test) in zip(
            ["input_tokens"],
            [true_input_tokens],
            [input_tokens],
        ):
            assert _test == true, f"{string} = {_test} != {true}"

        ##################################
        # 2. BertDataset
        ##################################
        data = BertDataset(
            encodings=encodings
        )  # data[j] = 4 torch tensors corresponding to EncodingKeys
        assert len(data) >= len(
            texts
        ), f"len(data) = {len(data)} < {len(texts)} = len(texts)"
        for i, (string, true) in enumerate(
            zip(
                ["input_ids", "attention_mask", "token_type_ids", "tag_ids"],
                [
                    true_input_ids,
                    true_attention_mask,
                    true_token_type_ids,
                    true_tag_ids,
                ],
            )
        ):
            for j in range(len(true)):
                assert torch.all(
                    torch.eq(data[j][i], true[j])
                ), f"{string} = {data[j][i]} != {true[j]}"


########################################################################################################################
########################################################################################################################
########################################################################################################################
class TestMisc:

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
            data_preprocessor._ensure_completeness_in_case_of_bio_tags(
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
        test_tag_list_ordered = order_tag_list(tag_list)
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
        test_tag_list = convert_tag_list_bio2plain(tag_list_bio)
        assert (
            test_tag_list == tag_list
        ), f"test_tag_list_ordered = {test_tag_list} != {tag_list}"


if __name__ == "__main__":
    test_csv = TestCsvReaderAndDataProcessor()
    test_csv.tests()

    test_input_examples_to_tensors = TestInputExamplesToTensorsAndBertDataset()
    test_input_examples_to_tensors.tests()

    test_misc = TestMisc()
    test_misc.test_ensure_completeness_in_case_of_bio_tags()
    test_misc.test_order_tag_list()
    test_misc.test_convert_tag_list_bio2plain()
