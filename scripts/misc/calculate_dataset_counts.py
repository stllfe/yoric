"""Helper script for getting Wikipedia `Ё` word counts.

Takes these CLI arguments:
    - the compressed wikipedia file path as an input.
    - the path to save the counts as a JSON file.
"""

import json
import logging
import multiprocessing as mp
import os
import sys

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


logging.getLogger().setLevel(logging.ERROR)

SAFE_DICT = yodict.get_safe()
NOT_SAFE_DICT = yodict.get_not_safe()

JOBSIZE = 100
NJOBS = os.cpu_count() or 1


def job(batch: list[WikiRecord]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for record in batch:
        text = utils.normalize_unicode(record.text)
        text = text.replace('\u0301', '')
        text = utils.normalize_wiki_text(text)
        for match in utils.WORDS_REGEX.finditer(text):
            word = match.group()
            if not utils.hasyeyo(word):
                continue
            if word in SAFE_DICT or word in NOT_SAFE_DICT:
                counter[word] = counter.get(word, 0) + 1
    return counter


def aggregate_job_results(jobs: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for counter in jobs:
        for word, count in counter.items():
            total[word] = total.get(word, 0) + count
    return total


def main() -> None:
    wiki = load_wiki(sys.argv[1])

    counts: dict[str, int] = {}
    with mp.Pool(NJOBS, initargs=(SAFE_DICT, NOT_SAFE_DICT)) as pool, tqdm(
        leave=True,
        desc='Calculating word coverage statistics',
        dynamic_ncols=True,
    ) as progress:
        jobs: list[list[WikiRecord]] = []
        for records in utils.batch(wiki, JOBSIZE):
            if len(jobs) < NJOBS:
                jobs.append(records)
            else:
                batch = list(pool.imap_unordered(job, jobs))
                jobs.clear()

                for word, count in aggregate_job_results(batch).items():
                    counts[word] = counts.get(word, 0) + count

                progress.update(NJOBS * JOBSIZE)
                progress.set_postfix_str(f'words={len(counts)}')

    with open(sys.argv[2], 'w', encoding='utf-8') as fp:
        json.dump(counts, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
