import pandas as pd
import argparse
from src.baseline_model import Baseline
from src.evaluate import evaluate_model 


def main(args: argparse.Namespace):
    df = pd.read_csv(args.data)
    df["yo_words"] = df["yo_words"].apply(eval)

    model = Baseline()
    evaluate_model(model=model, data=df, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data", 
        type=str,
        help="path to CSV file",
        default="data/test.csv"
    )
    main(parser.parse_args())