import pandas as pd
import argparse
from src.dict_model import DictModel
from src.evaluate import evaluate_model


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data)
    df["yo_words"] = df["yo_words"].apply(eval)

    evaluate_model(model=DictModel(safe_dict=args.safe_dict),
                   data=df,
                   verbose=True,
                   save_path=args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        help="path to CSV file",
        default="data/test.csv"
    )

    parser.add_argument(
        "--safe_dict",
        action=argparse.BooleanOptionalAction,
        help="Using safe dir or unsafe",
        default=False
    )

    parser.add_argument(
        "--save_path",
        help="path to save result in JSON fromat",
        required=False
    )

    main(parser.parse_args())
