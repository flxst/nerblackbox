"""
This script can be used to reproduce results of the paper
Adaptive Fine-Tuning of Transformer-Based Language Models for Named Entity Recognition
https://arxiv.org/abs/2202.02617

Note that in order to confirm the exact results, nerblackbox v0.0.12 (with pytorch-lightning==1.3.7) needs to be used!
Later versions of nerblackbox use later versions of pytorch-lightning which (seems to affect the random state and thus)
leads to similar but not identical results.

=== Run all experiments of a section: ===
python paper/script_paper.py --exp <experiment>
where <experiment> = main, appD, appG1, appG2, appG3, appH, appK, appL, scheme

=== Run subset of experiments of a section: ===
python paper/script_paper.py --exp <experiment> [--a <approach>]
                                                [--c <dataset-model combination>]
                                                [--x <dataset scaling factor>]
                                                [--xval <dataset scaling factor>]
where <approach> = original, stable, adaptive, hybrid
where <dataset-model combination> = I, II, III, IV, V
where <dataset scaling factor> = 0.005, 0.1, [..], 1.0
"""
import logging
import warnings
import argparse

from nerblackbox import NerBlackBox
from experiments.helper_functions import get_experiments, process_experiment_dict

logging.basicConfig(
    level=logging.WARNING
)  # basic setting that is mainly applied to mlflow's default logging
warnings.filterwarnings("ignore")


def main(exp: str, a: str = None, c: str = None, x: float = None, xval: float = None) -> None:
    """
    runs set of experiments defined by exp

    Args:
        exp: main, appD, appG1, appG2, appG3, appH, appK, appL
        a: filter annotation scheme: original, stable, adaptive, hybrid
        c: filter dataset-model combination: I, II, III, IV, V
        x: filter dataset scaling factor: 0.005, 0.01, ..
        xval: filter validation dataset scaling factor: 0.005, 0.01, ..
    """
    exp_filter = {k: v for k, v in {"a": a, "c": c, "x": x, "xval": xval}.items() if v is not None}
    experiments = get_experiments(exp, exp_filter)
    print(f"--> EXPERIMENTS: {len(experiments)}")

    nerbb = NerBlackBox()
    for i, experiment in enumerate(experiments):
        experiment_name = experiment.pop("experiment_name")
        experiment = process_experiment_dict(experiment)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(f"experiment {experiment_name} (#{i+1} out of {len(experiments)})")
        print(experiment)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        nerbb.run_experiment(experiment_name, **experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", required=True, type=str, help="main, appD, appG1, appG2, appG3, appH, appK, appL, scheme",
    )
    parser.add_argument(
        "--a", required=False, type=str, help="original, stable, adaptive, hybrid"
    )
    parser.add_argument(
        "--c", required=False, type=str, help="I, II, III, IV, V"
    )
    parser.add_argument(
        "--x", required=False, type=float, help="0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0"
    )
    parser.add_argument(
        "--xval", required=False, type=float, help="0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0"
    )
    _args = parser.parse_args()

    main(_args.exp, _args.a, _args.c, _args.x, _args.xval)
