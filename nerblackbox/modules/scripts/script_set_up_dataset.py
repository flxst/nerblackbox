import argparse
from nerblackbox.modules.datasets.formatter.auto_formatter import AutoFormatter
import mlflow


def main(args):
    """
    - I: get data for ner_dataset
    - II: write ner_tag_mapping.json file
    - III: format data
    - IV: resplit data
    - V: analyze and plot data
    --------------------------------------------------------------------------------
    :return: -
    """
    with mlflow.start_run(run_name="Default"):
        # formatter
        formatter = AutoFormatter.for_dataset(args.ner_dataset)

        formatter.create_directory()
        formatter.get_data(verbose=args.verbose)  # I: get_data
        formatter.create_ner_tag_mapping_json(
            modify=args.modify
        )  # II: create ner tag mapping
        formatter.format_data()  # III: format data
        formatter.resplit_data(val_fraction=args.val_fraction)  # IV: resplit data
        formatter.analyze_data()  # V: analyze data
        formatter.plot_data()  # V: analyze data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ner_dataset", required=True, type=str, help="e.g. swedish_ner_corpus"
    )
    parser.add_argument("--modify", type=bool, default=True)
    parser.add_argument("--val_fraction", type=float, default=0.3)
    parser.add_argument("--verbose", type=bool, default=False)
    _args = parser.parse_args()

    main(_args)
