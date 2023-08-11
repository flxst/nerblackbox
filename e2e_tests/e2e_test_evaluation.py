from nerblackbox import Store, Model
from datasets import load_dataset
from evaluate import evaluator
evaluator = evaluator("token-classification")

from os.path import abspath, isdir, join
import shutil
from utils import print_section_header, print_section_finish
from nerblackbox.api.model import EVALUATION_DICT

METRICS = ["precision", "recall", "f1"]

###################################################################
###################################################################
TEST_COMBINATIONS = [
    ["dslim/bert-base-NER", "conll2003", "test", 100],
    ["dslim/bert-base-NER", "conll2003", "val", 50],
    ["fhswf/bert_de_ner", "germeval_14", "test", 200],
]
###################################################################
###################################################################


def evaluate(_model: str, _dataset: str, split: str, number=None) -> EVALUATION_DICT:
    if split == "val":
        split = "validation"

    data = load_dataset(_dataset, split=split)

    if number is not None:
        data = data.select(range(number))

    evaluation_results = evaluator.compute(
        model_or_pipeline=_model,
        data=data,
        metric="seqeval",
    )

    # map huggingface evaluation dict format to nerblackbox evaluation dict format
    return {
        "micro": {
            "entity": {
                f"{metric}_seqeval": evaluation_results[f"overall_{metric}"]
                for metric in METRICS
            }
        }
    }


def test_evaluation(capsys):

    data_dir = abspath("./e2e_tests/e2e_test_evaluation_data")

    ################################################################################################################
    print_section_header(f"0. Store.set_path([..])")
    Store.set_path(data_dir)
    print_section_finish()

    ################################################################################################################
    if isdir(data_dir):
        shutil.rmtree(data_dir)
        print(f"> removed {data_dir}\n")

    try:
        ################################################################################################################
        print_section_header(f"1. Store.create()")
        Store.create()
        assert isdir(data_dir), f"ERROR! data_dir = {data_dir} does not exist."
        for subdirectory in ["datasets", "experiment_configs", "pretrained_models", "results"]:
            assert isdir(join(data_dir, subdirectory)), \
                f"ERROR! data_dir/subdirectory = {join(data_dir, subdirectory)} does not exist."
        print_section_finish()

        for i, test_combination in enumerate(TEST_COMBINATIONS):
            ############################################################################################################
            model, dataset, phase, number = test_combination
            print_section_header(f"> Test #{i+1}: model={model}, dataset={dataset}, phase={phase}, number={number}")

            # nerblackbox
            model_huggingface = Model.from_huggingface(model)
            evaluation_dict_nerblackbox = model_huggingface.evaluate_on_dataset(dataset,
                                                                                # "huggingface",
                                                                                phase=phase,
                                                                                number=number,
                                                                                rounded_decimals=None)

            # huggingface evaluate
            evaluation_dict_huggingface = evaluate(model,
                                                   dataset,
                                                   phase,
                                                   number)

            print()
            print("evaluation_dict_nerblackbox:", evaluation_dict_nerblackbox)
            print("evaluation_dict_huggingface:", evaluation_dict_huggingface)
            print()

            for metric in METRICS:
                result_nerblackbox = evaluation_dict_nerblackbox["micro"]["entity"][f"{metric}_seqeval"]
                result_huggingface = evaluation_dict_huggingface["micro"]["entity"][f"{metric}_seqeval"]
                assert result_nerblackbox == result_huggingface, \
                    f"ERROR! metric = {metric}_seqeval: nerblackbox = {result_nerblackbox}, " \
                    f"huggingface evaluate = {result_huggingface}"

            print_section_finish()

    except Exception as e:
        raise Exception(e)
    finally:
        # stdout & stderr to files
        out, err = capsys.readouterr()
        with open(join(data_dir, "err.txt"), "w") as f:
            f.write(err)
        with open(join(data_dir, "out.txt"), "w") as f:
            f.write(out)
