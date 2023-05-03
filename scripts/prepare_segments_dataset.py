"""A script for initial preprocessing and cleaning of `Ё` text segments."""

import argparse
import csv

from typing import List

import numpy as np
import pandas as pd

import humanfriendly as hf

from src import utils


def to_dataframe(texts: List[str], sep: str = utils.SEPARATOR) -> pd.DataFrame:
    """Writes raw lines of a text file to a :class:`DataFrame` with metadata (if available)."""

    rows = []
    for text in texts:
        l, s, r = text.strip().replace('""', '"').partition(sep)
        if not s:
            rows.append((None, l))
        else:
            rows.append((l, r))

    data = pd.DataFrame(rows, columns=('record_id', 'text'))
    if data.record_id.any():
        data.record_id = data.record_id.bfill()
    return data


def clean_by_iqr(
    data: pd.DataFrame,
    column: str,
    q1: float = 0.25,
    q3: float = 0.75,
    offset: float = 1.5,
) -> pd.DataFrame:
    """Applies IQR cleaning by the column values."""

    lo = data[column].quantile(q1)
    hi = data[column].quantile(q3)
    dt = offset * (hi - lo)

    return data[data[column].between(lo - dt, hi + dt, inclusive='both')]


def enrich_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Renames columns and adds new ones.

    Final schema:
        - `record_id` — Wikipedia record ID
        - `yo_text` — original text with `Ё`
        - `ye_text` — text with `Ё` -> `Е` replaced
        - `text_length` — text length in characters
        - `text_loglen` — the logarithm of text length
        - `yo_words` — list of tuples of all `Ё` substrings in the given text
    """

    new = pd.DataFrame()

    new['record_id'] = data.record_id.copy()
    new['yo_text'] = data.text.copy()
    new['ye_text'] = data.text.apply(utils.yeficate)

    new['text_length'] = data.text.apply(len)
    new['text_loglen'] = new.text_length.apply(np.log)

    new['yo_words'] = data.text.apply(utils.get_not_safe_yo_substrings)

    return new


def main(args: argparse.Namespace) -> None:
    print(f'Original file size: {hf.format_size(utils.get_filesize(args.data_path))}')

    with open(args.data_path, encoding='utf-8') as file:
        texts = file.readlines()

    data = to_dataframe(texts)
    print(f'Initial number of rows: {len(data)}')

    data = enrich_columns(data)
    data = clean_by_iqr(data, column='text_loglen')
    data = data[data.text_length.lt(args.max_text_length)]

    data.to_csv(args.save_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f'Final number of rows: {len(data)}')
    print(f'CSV file successfully saved: {args.save_path}')
    print(f'Final file size: {hf.format_size(utils.get_filesize(args.save_path))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--data-path',
        help='a path to a raw wiki segments TXT file',
        default='data/ruwiki-yo-segments.txt'
    )
    parser.add_argument(
        '-o', '--save-path',
        help='a filepath to save the cleaned segments CSV file',
        default='data/ruwiki-yo-segments-preprocessed.csv'
    )
    parser.add_argument(
        '--max-text-length',
        metavar='INT',
        type=int,
        default=220,
        help='a maximum number of characters in the cleaned segments'
    )
    main(parser.parse_args())
