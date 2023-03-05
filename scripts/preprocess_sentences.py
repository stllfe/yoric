"""A script for initial preprocessing/cleaning of `Ё` sentences."""

import argparse
import os
import multiprocessing as mp

import humanfriendly as hf
from tqdm import tqdm
from src import utils


def get_filesize(filepath: str) -> int:
    """Returns filesize in bytes."""

    return os.stat(filepath).st_size


def main(args: argparse.Namespace):
    with open(args.data_path, encoding='utf-8') as file:
        data = file.readlines()

    print('Opened initial data.')
    print(f'Filesize is {hf.format_size(get_filesize(args.data_path))}')

    with mp.Pool(8) as pool, tqdm(
        pool.imap_unordered(utils.normalize_wiki_text, data),
        total=len(data),
        desc='Cleaning `Ё` sentences'
    ) as progress:
        new_data = list(progress)

    with open(args.save_path, 'w', encoding='utf-8') as file:
        for sentence in filter(bool, new_data):
            file.write(sentence + '\n')

    print('Saved results to a new file.')
    get_filesize(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-d', '--data-path',
        help='a path to a raw wiki sentences TXT file',
        default='data/ruwiki-yo-sentences.txt'
    )
    parser.add_argument(
        '-f', '--save-path',
        help='a filepath to save the cleaned sentences',
        default='data/ruwiki-yo-sentences-preprocessed.txt'
    )
    parser.add_argument(
        '-j', '--njobs',
        metavar='INT',
        type=int,
        default=4,
        help='a number of parallel jobs',
    )
    main(parser.parse_args())
