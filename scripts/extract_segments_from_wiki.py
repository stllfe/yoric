"""A script that extracts the segments with `Ё` letter from the Russian Wikipedia dump."""

import argparse
import logging
import multiprocessing as mp
import os

from functools import partial
from multiprocessing.pool import Pool
from typing import Protocol, TYPE_CHECKING

from corus.sources.wiki import load_wiki
from tqdm import tqdm

from yogurt import utils
from yogurt import yodict


if TYPE_CHECKING:

    class WikiRecord(Protocol):
        id: int
        title: str
        text: str

else:
    from corus.sources.wiki import WikiRecord


# suppress warnings from Wiki extractor
logging.getLogger().setLevel(logging.ERROR)

NJOBS = os.cpu_count() or 1

NOT_SAFE_DICT = yodict.get_not_safe()


def has_not_safe_words(text: str) -> bool:
    for match in utils.WORDS_REGEX.finditer(text):
        word = match.group()
        if not utils.hasyeyo(word):
            continue
        if word in NOT_SAFE_DICT:
            return True
    return False


def job(
    records: list[WikiRecord], max_text_length: int, id_sep: str = utils.SEPARATOR
) -> list[str]:
    segments: list[str] = []
    for record in records:
        text = utils.normalize_unicode(record.text)
        text = text.replace('\u0301', '')
        text = utils.normalize_wiki_text(text)
        for segment in utils.extract_unique_yo_segments(text):
            if len(segment) > max_text_length:
                continue
            if has_not_safe_words(segment):
                segments.append(f'{record.id}{id_sep}{segment}')
    return segments


def run_jobs(pool: Pool, args: argparse.Namespace, jobs: list[list[WikiRecord]]) -> list[str]:
    func = partial(job, max_text_length=args.max_text_length)
    results = pool.imap_unordered(func, jobs)
    return sum(results, [])


def main(args: argparse.Namespace) -> None:
    assert (
        args.num_segments is None or args.num_segments > 0
    ), '`num_segments` should be a positive integer!'
    wiki = load_wiki(args.wiki_path)

    segments = []
    with mp.Pool(args.njobs) as pool, tqdm(
        total=args.num_segments,
        leave=True,
        desc='Extracting `Ё` segments from wiki records',
        dynamic_ncols=True,
    ) as progress:
        jobs: list[list[WikiRecord]] = []
        for records in utils.batch(wiki, args.jobsize):
            if len(jobs) < args.njobs:
                jobs.append(records)
            else:
                found = run_jobs(pool, args, jobs)
                progress.update(len(found))
                segments.extend(found)
                jobs.clear()

            if args.num_segments and len(segments) >= args.num_segments:
                segments = segments[: args.num_segments]
                progress.update()
                progress.close()
                break

    with open(args.save_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(segment + '\n')

    print(f'File saved to: {args.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i',
        '--wiki-path',
        help='a path to wiki dump',
        default='data/ruwiki-latest-pages-articles.xml.bz2',
    )
    parser.add_argument(
        '-o',
        '--save-path',
        help='a filepath to save the segments',
        default='data/ruwiki-yo-segments.txt',
    )
    parser.add_argument(
        '-j',
        '--njobs',
        metavar='INT',
        type=int,
        default=NJOBS,
        help='a number of parallel jobs',
    )
    parser.add_argument(
        '-s',
        '--jobsize',
        metavar='INT',
        type=int,
        default=100,
        help='a number of documents for a single job',
    )
    parser.add_argument(
        '-n',
        '--num-segments',
        metavar='INT',
        type=int,
        default=None,
        help='a hard limit of segments to gather',
    )
    parser.add_argument(
        '--max-text-length',
        metavar='INT',
        type=int,
        default=220,
        help='a maximum number of characters in the cleaned segments',
    )
    main(parser.parse_args())
