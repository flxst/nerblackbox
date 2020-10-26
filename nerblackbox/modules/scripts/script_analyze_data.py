import argparse
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter
import mlflow


def main(args):
    """
    - V: analyze data
      . read reshaped text files
      . analyze data and output class distribution
    --------------------------------------------------------------------------------
    :return: -
    """
    with mlflow.start_run(run_name="Default"):
        formatter = AutoFormatter.for_dataset(args.ner_dataset)
        formatter.analyze_data()  # IV: analyze data
        formatter.plot_data()  # IV: analyze data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ner_dataset", required=True, type=str, help="e.g. swedish_ner_corpus"
    )
    parser.add_argument("--verbose", type=bool, default=False)
    _args = parser.parse_args()

    main(_args)
