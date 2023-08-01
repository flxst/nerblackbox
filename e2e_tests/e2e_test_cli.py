import os
import subprocess
from os.path import abspath, isdir, join
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
    if isdir(data_dir):
        shutil.rmtree(data_dir)
        print(f"> removed {data_dir}\n")
    os.makedirs(data_dir)

    try:
        ################################################################################################################
        print_section_header(f"0. nerblackbox --help")
        run_cli("nerblackbox --help")
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
