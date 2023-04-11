"""A script that extracts the segments with `Ё` letter from the Russian Wikipedia dump."""

import argparse
import logging
import multiprocessing as mp

from typing import List

from corus.sources.wiki import WikiRecord, load_wiki
from tqdm import tqdm

from src import utils


# suppress warnings from Wiki extractor
logging.getLogger().setLevel(logging.ERROR)


def job(records: List[WikiRecord], id_sep: str = utils.SEPARATOR) -> List[str]:
    segments = []
    for record in records:
        normalized = utils.normalize_wiki_text(record.text)
        for segment in utils.extract_unique_yo_segments(normalized):
            segments.append(f'{record.id}{id_sep}{segment}')
    return segments


def aggregate_job_results(pool: mp.Pool, jobs: List[List[WikiRecord]]) -> List[str]:
    results = pool.imap_unordered(job, jobs)
    return sum(results, [])


def main(args: argparse.Namespace):
    assert args.num_segments is None or args.num_segments > 0, (
        '`num_segments` should be a positive integer!'
    )
    wiki = load_wiki(args.wiki_path)

    segments = []
    with mp.Pool(args.njobs) as pool, tqdm(
        total=args.num_segments,
        leave=True,
        desc='Extracting `Ё` segments from wiki records',
        dynamic_ncols=True,
    ) as progress:
        jobs = []
        for records in utils.batch(wiki, args.jobsize):
            if len(jobs) < args.njobs:
                jobs.append(records)
            else:
                found = aggregate_job_results(pool, jobs)
                progress.update(len(found))
                segments.extend(found)
                jobs.clear()

            if args.num_segments and len(segments) >= args.num_segments:
                segments = segments[:args.num_segments]
                progress.update()
                progress.close()
                break

    with open(args.save_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(segment + '\n')

    print(f'File saved to: {args.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--wiki-path',
        help='a path to wiki dump',
        default='data/ruwiki-latest-pages-articles.xml.bz2'
    )
    parser.add_argument(
        '-o', '--save-path',
        help='a filepath to save the segments',
        default='data/ruwiki-yo-segments.txt'
    )
    parser.add_argument(
        '-j', '--njobs',
        metavar='INT',
        type=int,
        default=4,
        help='a number of parallel jobs',
    )
    parser.add_argument(
        '-s', '--jobsize',
        metavar='INT',
        type=int,
        default=50,
        help='a number of documents for a single job',
    )
    parser.add_argument(
        '-n', '--num-segments',
        metavar='INT',
        type=int,
        default=None,
        help='a hard limit of segments to gather'
    )
    main(parser.parse_args())
