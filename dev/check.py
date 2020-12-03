
import argparse
import subprocess

from os.path import abspath, dirname
BASE_DIR = abspath(dirname(dirname(abspath(__file__))))


def main(args):
    assert args.black is not None or args.pylint or args.mypy or args.coverage, \
        f'need to specify --black|--pylint|--mypy|--coverage'

    commands_list = [f'cd {BASE_DIR}/dev']
    if args.black is not None:
        if args.black == 'check':
            command = f'black --check {BASE_DIR}/nerblackbox'
        elif args.black == 'diff':
            command = f'black --diff --color {BASE_DIR}/nerblackbox'
        elif args.black == 'convert':
            command = f'black {BASE_DIR}/nerblackbox'
        else:
            raise Exception(f'need to specify --black check|diff|convert')
        commands_list.append(command)
    if args.pylint:
        command = f'pylint {BASE_DIR}/nerblackbox'
        commands_list.append(command)
    if args.mypy:
        # --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs
        command = f'mypy --config-file=pyproject.toml {BASE_DIR}/nerblackbox'
        commands_list.append(command)
    if args.coverage:
        command = f'coverage run --source=nerblackbox ' \
                  f'-m pytest {BASE_DIR}/nerblackbox; ' \
                  f'coverage html; ' \
                  f'coverage report'
        commands_list.append(command)

    commands = ';'.join(commands_list)
    print(f'### {commands}')
    subprocess.run(commands, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--black', type=str, help='check, diff or convert')
    parser.add_argument('--pylint', action='store_true', default=False)
    parser.add_argument('--mypy', action='store_true', default=False)
    parser.add_argument('--coverage', action='store_true', default=False)
    _args = parser.parse_args()

    main(_args)
