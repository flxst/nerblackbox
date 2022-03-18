from nerblackbox import NerBlackBox, NerModelPredict
from os.path import abspath, isdir, join, isfile
import shutil
from utils import print_section_header, print_section_finish


def test_api(capsys):

    data_dir = abspath("./e2e_tests/e2e_test_api_data")
    experiment_name = "e2e_test_api_experiment"
    model_name = "mrm8488/electricidad-base-discriminator"
    dataset_name = "ehealth_kd"
    dataset_suffix = "jsonl"
    input_text = "La hepatitis es una inflamación del hígado."

    if isdir(data_dir):
        shutil.rmtree(data_dir)
        print(f"> removed {data_dir}\n")

    ####################################################################################################################
    print_section_header(f"1. nerbb = NerBlackBox(data_dir=data_dir) with data_dir = {data_dir}")
    nerbb = NerBlackBox(data_dir=data_dir)
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"2. nerbb.init()")
    nerbb.init()
    assert isdir(data_dir), f"ERROR! data_dir = {data_dir} does not exist."
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"3. nerbb.set_up_dataset(dataset) with dataset = {dataset_name}")
    nerbb.set_up_dataset(dataset_name)
    for phase in ["train", "val", "test"]:
        file_path = join(data_dir, "datasets", dataset_name, f"{phase}.{dataset_suffix}")
        assert isfile(file_path), f"ERROR! file = {file_path} does not exist."
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"4. nerbb.run_experiment()")
    nerbb.run_experiment(experiment_name,
                         model=model_name,
                         dataset=dataset_name,
                         lr_warmup_epochs=0,
                         max_epochs=3,  # 3
                         prune_ratio_train=0.2,  # 0.5
                         prune_ratio_val=0.01,   # 0.01
                         prune_ratio_test=0.2,   # 0.5
                         multiple_runs=1,
                         from_preset="original",
                         )
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"5. nerbb.get_experiment_results()")
    experiment_results = nerbb.get_experiment_results(experiment_name)
    assert isinstance(experiment_results, list), f"ERROR! experiment_results is not a list."
    assert len(experiment_results) == 1, f"ERROR! len(experiment_results) = {len(experiment_results)} should be 1."
    for attribute in ["best_single_run"]:
        assert hasattr(experiment_results[0], attribute), \
            f"ERROR! experiment_results[0] does not have attribute = {attribute}"
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"6. nerbb.get_model_from_experiment()")
    ner_model_predict = nerbb.get_model_from_experiment(experiment_name)
    assert isinstance(ner_model_predict, NerModelPredict), \
        f"ERROR! ner_model_predict is of type {type(ner_model_predict)} but should be NerModelPredict"
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"7. nerbb.predict()")
    predictions1 = nerbb.predict(experiment_name, input_text)
    print(predictions1)
    print_section_finish()

    ####################################################################################################################
    print_section_header(f"8. ner_model_predict.predict()")
    predictions2 = ner_model_predict.predict(input_text)
    print(predictions2)
    print_section_finish()

    assert predictions1 == predictions2, \
        f"ERROR! predictions from nerbb.predict() are different from ner_model_predit.predict()"

    ####################################################################################################################
    ####################################################################################################################
    # stdout & stderr to files
    out, err = capsys.readouterr()
    open(join(data_dir, "err.txt"), "w").write(err)
    open(join(data_dir, "out.txt"), "w").write(out)
