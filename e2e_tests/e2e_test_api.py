from nerblackbox import Store, Dataset, Experiment, Model
from nerblackbox.modules.experiment_results import ExperimentResults
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

        ################################################################################################################
        print_section_header(f"2. dataset = Dataset({dataset_name}).set_up()")
        dataset = Dataset(name=dataset_name, source="HF")
        dataset.set_up()
        for phase in ["train", "val", "test"]:
            file_path = join(data_dir, "datasets", dataset_name, f"{phase}.{dataset_suffix}")
            assert isfile(file_path), f"ERROR! file = {file_path} does not exist."
        print_section_finish()

        ################################################################################################################
        print_section_header(f"3. experiment = Experiment([..]).run()")
        experiment = Experiment(
            experiment_name,
            model=model_name,
            dataset=dataset_name,
            lr_warmup_epochs=0,
            max_epochs=3,  # 3
            prune_ratio_train=0.2,  # 0.5
            prune_ratio_val=0.01,  # 0.01
            prune_ratio_test=0.2,  # 0.5
            multiple_runs=1,
            from_preset="original",
        )
        experiment.run()
        print_section_finish()

        ################################################################################################################
        print_section_header(f"4. experiment.results")
        assert isinstance(experiment.results, ExperimentResults), \
            f"ERROR! experiment.results is not an instance of ExperimentResults."
        for attribute in ["best_single_run"]:
            assert hasattr(experiment.results, attribute), \
                f"ERROR! experiment.results does not have attribute = {attribute}"

        for average in [False, True]:
            score = experiment.get_result(metric="f1", level="entity", label="micro", phase="test", average=average)
            print(f"score (average={average}) = {score}")
            assert isinstance(score, str), \
                f"ERROR! experiment.get_result() did not return a str for average = {average}."
        print_section_finish()

        ################################################################################################################
        print_section_header(f"5. model = Model.from_experiment()")
        model = Model.from_experiment(experiment_name)
        assert isinstance(model, Model), \
            f"ERROR! model is of type {type(model)} but should be Model"
        print_section_finish()

        ################################################################################################################
        print_section_header(f"6. model.predict()")
        predictions = model.predict(input_text)
        print(predictions)
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
