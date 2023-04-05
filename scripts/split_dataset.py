"""A script for generating train-test split of `Ё` text segments."""

import argparse
import csv

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


SEED = 42


def main(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.csv_path, quoting=csv.QUOTE_NONNUMERIC)
    print(f'Total number of rows: {len(data)}')

    train, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=SEED,
        shuffle=True,
    )
    print(f'Final split: train={len(train)} test={len(test)}')
    print(f'Test size: {args.test_size}\n')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    train.to_csv(save_dir / 'train.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    test.to_csv(save_dir / 'test.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f'CSV files successfully saved to: {save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--csv-path',
        help='a path to a preprocessed wiki segments CSV file',
        default='data/ruwiki-yo-segments-preprocessed.csv'
    )
    parser.add_argument(
        '-O', '--save-dir',
        help='a dirpath to save the splitted segments CSV files (train, test)',
        default='data'
    )
    parser.add_argument(
        '--test-size',
        metavar='FLOAT',
        type=float,
        help='a test fraction (or an absolute number of samples)',
        default=0.25,
    )
    main(parser.parse_args())
