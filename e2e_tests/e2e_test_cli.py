import subprocess
from os.path import abspath, isdir, join, isfile
import shutil
from utils import print_section_header, print_section_finish


def run_cli(bash_cmd: str) -> None:
    try:
        result = subprocess.run(bash_cmd, shell=True, check=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        print(result)
    except subprocess.CalledProcessError as e:
        raise Exception(e)


def test_cli(capsys):

    data_dir = abspath("./e2e_tests/e2e_test_cli_data")
    experiment_name = "e2e_test_cli_experiment"
    if 1:
        model_name = "mrm8488/electricidad-base-discriminator"
        dataset_name = "ehealth_kd"
        dataset_suffix = "jsonl"
        input_text = "La hepatitis es una inflamación del hígado."
        experiment_config_lines = [
            "[dataset]",
            f"dataset_name = {dataset_name}",
            "prune_ratio_train = 0.2",
            "prune_ratio_val = 0.01",
            "prune_ratio_test = 0.2",
            "",
            "[model]",
            f"pretrained_model_name = {model_name}",
            "",
            "[settings]",
            "multiple_runs = 1",
            "",
            "[hparams]",
            "max_epochs = 3",
            "lr_warmup_epochs = 0",
            "",
            "[runA]",
        ]
    else:
        model_name = "KB/bert-base-swedish-cased"
        dataset_name = "swedish_ner_corpus"
        dataset_suffix = "csv"
        input_text = "Ricardo Sa Pinto tar över ."
        experiment_config_lines = [
            "[dataset]",
            f"dataset_name = {dataset_name}",
            "prune_ratio_train = 0.01",
            "prune_ratio_val = 0.01",
            "prune_ratio_test = 0.01",
            "",
            "[model]",
            f"pretrained_model_name = {model_name}",
            "",
            "[settings]",
            "multiple_runs = 1",
            "",
            "[hparams]",
            "max_epochs = 1",
            "early_stopping = False",
            "lr_schedule = linear",
            "lr_warmup_epochs = 0",
            "",
            "[runA]",
        ]

    if isdir(data_dir):
        shutil.rmtree(data_dir)
        print(f"> removed {data_dir}\n")

    try:
        ################################################################################################################
        print_section_header(f"0. nerbb --help")
        run_cli("nerbb --help")
        print_section_finish()

        ################################################################################################################
        print_section_header(f"1. nerbb create")
        run_cli(f"nerbb --store_dir {data_dir} create")
        assert isdir(data_dir), f"ERROR! data_dir = {data_dir} does not exist."
        for subdirectory in ["datasets", "experiment_configs", "pretrained_models", "results"]:
            subdirectory_path = join(data_dir, subdirectory)
            assert isdir(subdirectory_path), f"ERROR! subdirectory = {subdirectory_path} does not exist."
        print_section_finish()

        ################################################################################################################
        print_section_header(f"2. nerbb set_up_dataset {dataset_name}")
        run_cli(f"nerbb --store_dir {data_dir} set_up_dataset {dataset_name}")
        for phase in ["train", "val", "test"]:
            file_path = join(data_dir, "datasets", dataset_name, f"{phase}.{dataset_suffix}")
            assert isfile(file_path), f"ERROR! file = {file_path} does not exist."
        print_section_finish()

        ################################################################################################################
        print_section_header(f"X. create experiment config based on e2e_test_api_experiment.ini")
        experiment_config_lines = [line + "\n" for line in experiment_config_lines]
        file_path_e2e_test_cli_experiment = join(data_dir, "experiment_configs", "e2e_test_cli_experiment.ini")
        with open(file_path_e2e_test_cli_experiment, "w") as f:
            f.writelines(experiment_config_lines)
        assert isfile(file_path_e2e_test_cli_experiment), \
            f"ERROR! file = {file_path_e2e_test_cli_experiment} does not exist. run e2e_test_api.py first!"
        print_section_finish()

        ################################################################################################################
        print_section_header(f"3. nerbb run_experiment")
        run_cli(f"nerbb --store_dir {data_dir} run_experiment {experiment_name}")
        print_section_finish()

        ################################################################################################################
        print_section_header(f"4. nerbb get_experiment_results")
        run_cli(f"nerbb --store_dir {data_dir} get_experiment_results {experiment_name}")
        print_section_finish()

        ################################################################################################################
        print_section_header(f"6. nerbb.predict()")
        run_cli(f"nerbb --store_dir {data_dir} predict {experiment_name} '{input_text}'")
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
