"""A script for `Ё` texts preprocessing and generating a dataset markup."""

import argparse
import enum
import multiprocessing as mp
import operator as op
import os
import random
import warnings

from collections import Counter
from collections import deque
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Optional, Union

import humanfriendly as hf
import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from numpy.typing import NDArray
from tqdm import tqdm

from yogurt import data
from yogurt import utils
from yogurt import yodict


SEED = 42

NOT_SAFE_DICT = yodict.get_not_safe()
SAFE_DICT = yodict.get_safe()

MIN_COUNT = 2
X_LIMIT = 10
TEST_FRACTION = 0.25

NJOBS = os.cpu_count() or 1
CHUNKSIZE = 100

VOCAB_FILENAME = 'vocab.txt'
MARKUPS_FILENAME = 'markups.jsonl.bz2'


class SampleStrategy(str, enum.Enum):
    NONE = 'none'
    DOWNSAMPLE = 'downsample'
    UPPERLIMIT = 'upperlimit'

    def __str__(self) -> str:
        return self.value


def read_segments(
    filepath: Union[str, Path], id_sep: str = utils.SEPARATOR
) -> tuple[list[str], list[Optional[int]]]:
    """Read raw lines of a segments text file and returns texts and record IDs (if provided)."""

    record_ids: list[Optional[int]] = []
    texts: list[str] = []
    with open(filepath, encoding='utf-8') as fp:
        for line in fp:
            l, s, r = line.strip().replace('""', '"').partition(id_sep)
            if not s:
                texts.append(l)
                record_ids.append(None)
            else:
                texts.append(r)
                record_ids.append(int(l))

    return texts, record_ids


def extract_not_safe_word_counts(texts: list[str]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for text in texts:
        for match in utils.WORDS_REGEX.finditer(text):
            word = match.group()
            if not utils.hasyeyo(word):
                continue
            if word in NOT_SAFE_DICT and word not in SAFE_DICT:
                counter[word] = counter.get(word, 0) + 1
    return counter


def get_counts_stats(counts: dict[str, int], min_count: int = 1) -> pd.DataFrame:
    index: dict[tuple[str, str], int] = {}
    for word, count in counts.items():
        word = word.lower()
        key = utils.yeficate(word)
        index[(key, word)] = index.get((key, word), 0) + count
    stats = pd.DataFrame(
        [(key, word, count) for (key, word), count in index.items()], columns=['key', 'word', 'cnt']
    )
    stats = stats.groupby('key').filter(
        lambda x: (x.word.count() == 2) & (x.cnt.min() >= min_count)
    )
    stats['cnt_diff'] = stats.groupby('key').cnt.transform(lambda x: x.max() - x.min())
    stats['cnt_divs'] = stats.groupby('key').cnt.transform(lambda x: x.max() / x.min())
    return stats.sort_values(['key', 'cnt'], ignore_index=True)


def get_downsampled_counts(stats: pd.DataFrame) -> dict[str, int]:
    counts = stats.copy()
    counts['target'] = counts.cnt.copy()
    counts['target'] = counts.groupby('key').cnt.transform(lambda x: x.min())
    counts.set_index('word', inplace=True)
    return counts['target'].to_dict()


def get_upperlimited_counts(stats: pd.DataFrame, limit: int = X_LIMIT) -> dict[str, int]:
    counts = stats.copy()
    counts['target'] = counts.cnt.copy()
    counts['target'] = counts.groupby('key').cnt.transform(
        lambda x: x.apply(lambda y: y if y == x.min() else min(y, x.min() * limit))
    )
    counts.set_index('word', inplace=True)
    return counts['target'].to_dict()


def sample_texts_by_word_counts(
    texts: list[str], counts: dict[str, int], random_state: int = 42, shuffle: bool = True
) -> list[str]:
    """Samples texts from the given collection till
    it's possible to satisfy the target count per word."""

    target = counts.copy()
    counts = {}

    if shuffle:
        random.seed(random_state)
        texts = texts.copy()
        random.shuffle(texts)

    sampled = []
    current = 0
    total = sum(target.values())
    queue = deque(texts)

    with tqdm(total=total, desc='Sampling texts', leave=False) as pbar:
        while current < total and queue:
            x_text = queue.popleft()
            x_counts = extract_not_safe_word_counts([x_text])
            for word, count in x_counts.items():
                word = word.lower()
                curr_count = counts.get(word, 0)
                if word not in target:
                    continue
                if curr_count + count > target[word]:
                    break
                counts[word] = curr_count + count
                current += count
                pbar.update(count)
            else:
                sampled.append(x_text)
            pbar.set_postfix(queue=f'{len(queue):d}')

    tqdm.write(f'Sampling reached {current / total * 100:.2f}% of target counts!')
    return sampled


def extract_yeyo_markup(text: str, vocab: data.Vocab) -> data.YeYoMarkup:
    yes = utils.get_ye_substrings(text, NOT_SAFE_DICT)
    yos = utils.get_yo_substrings(text, NOT_SAFE_DICT)
    spans = sorted(yes + yos, key=op.itemgetter(1))
    words, targets = [], []
    for start, end in spans:
        word = text[start:end]
        if word.lower() in vocab:
            words.append(word.lower())
        if utils.hasyo(word):
            targets.append(1)
        else:
            targets.append(0)
    labels = list(map(vocab.get_label, words))
    return data.YeYoMarkup(text, spans, labels, targets)


def extract_yeyo_markups_parallel(
    texts: Iterable[str], vocab: data.Vocab, njobs: int = NJOBS
) -> list[data.YeYoMarkup]:
    texts = list(texts)
    with mp.Pool(processes=njobs) as pool:
        func = partial(extract_yeyo_markup, vocab=vocab)
        jobs = pool.imap_unordered(func, texts, chunksize=CHUNKSIZE)
        with tqdm(
            jobs, total=len(texts), desc=f'Extracting markup (njobs={njobs})', leave=False
        ) as pbar:
            markups = list(pbar)
    return markups


def get_targets_counter(markups: Iterable[data.YeYoMarkup]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for m in markups:
        counter.update(m.targets)
    return counter


def stratify_by_words(
    markups: Iterable[data.YeYoMarkup], test_size: float, random_state: int = SEED
) -> tuple[NDArray, NDArray]:
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    X = list(markups)
    N, M = len(X), max(len(m.labels) for m in markups)

    pad_value = -100
    Y = np.ones((N, M), np.int16) * pad_value

    for i, sample in enumerate(X):
        for j, value in enumerate(sample.labels):
            Y[i, j] = value

    print(f'Label encoder matrix size: {N}x{M}')
    return next(splitter.split(np.zeros(N), Y))  # type: ignore


def main(args: argparse.Namespace) -> None:
    """The sampling pipeline."""

    if not args.save_dir.is_dir():
        raise NotADirectoryError(args.save_dir)

    texts, _ = read_segments(args.data_path)

    print('Data loaded. Extracting word counts...')
    counts = extract_not_safe_word_counts(texts)

    total_not_safe = len(set(NOT_SAFE_DICT) - set(SAFE_DICT))
    print(
        f'Unique words:  {len(counts)} ({len(counts) / total_not_safe * 100:.1f}% of not-safe dictionary)'
    )

    # todo: ugly code next...
    print(f'Using sample strategy: {args.strategy}')
    if args.strategy is SampleStrategy.NONE:
        resampled_texts = texts
    else:
        stats = get_counts_stats(counts, min_count=args.min_word_count)

        if args.strategy is SampleStrategy.DOWNSAMPLE:
            counts = get_downsampled_counts(stats)
        elif args.strategy is SampleStrategy.UPPERLIMIT:
            assert args.x_limit > 0, f'Wrong value for --x-limit: {args.x_limit}'
            counts = get_upperlimited_counts(stats, args.x_limit)
        else:
            raise NotImplementedError(args.strategy)

        print('Target counts calculated.')
        resampled_texts = sample_texts_by_word_counts(texts, counts)

    print('Calculating stats on the sampled texts...')
    counts = extract_not_safe_word_counts(resampled_texts)
    uwords = set(map(str.lower, counts.keys()))

    vocab = data.Vocab(uwords)
    print(f'Vocab size: {len(vocab)}')

    markups = extract_yeyo_markups_parallel(resampled_texts, vocab, njobs=args.njobs)

    print(f'Extracted markups: {len(markups)}')
    markups = list(filter(lambda m: len(m.labels) > 0, markups))
    print(f'Markups after filtering: {len(markups)}')

    stats = get_counts_stats(counts, min_count=int(np.ceil(1 / args.test_size)))

    print('Stratifying the dataset splits by words...')
    train_index, test_index = stratify_by_words(markups, test_size=args.test_size)
    print('Indices ready. Filtering common words...')

    train_indexer = op.itemgetter(*train_index)
    test_indexer = op.itemgetter(*test_index)

    # most likely some sentences took more than one word, so we have to filter again
    train_texts = train_indexer(resampled_texts)
    test_texts = test_indexer(resampled_texts)

    train_counts = get_counts_stats(extract_not_safe_word_counts(train_texts))
    test_counts = get_counts_stats(extract_not_safe_word_counts(test_texts))

    common_keys = set(train_counts.key.values) & set(test_counts.key.values)
    unique_keys = set(stats.key.unique())

    if (len(unique_keys) - len(common_keys)) / len(unique_keys) > 0.1:
        warnings.warn(
            'More than 10% difference between the estimated and the actual key counts!',
            UserWarning,
            stacklevel=2,
        )

    # words could have been included on the markup extraction step, since we only used the NOT_SAFE dictionary there
    # at the same time, the SAFE and NOT_SAFE dictionary have an intersection of words -> must be discarded
    common_keys = {word for word in common_keys if word not in SAFE_DICT}
    print(f'Unique common keys: {len(common_keys)}')

    def has_common_keys(markup: data.YeYoMarkup) -> bool:
        words = set(map(utils.yeficate, map(vocab.get_word, markup.labels)))
        return bool(words & common_keys)

    train_markups = list(filter(has_common_keys, train_indexer(markups)))
    test_markups = list(filter(has_common_keys, test_indexer(markups)))

    print(f'Train targets: {get_targets_counter(train_markups)}')
    print(f'Test targets: {get_targets_counter(test_markups)}')

    train_markups_path = args.save_dir / '-'.join(('train', args.markups_filename))
    test_markups_path = args.save_dir / '-'.join(('test', args.markups_filename))

    data.save_markups(train_markups, train_markups_path, compress=args.compress)
    data.save_markups(test_markups, test_markups_path, compress=args.compress)
    vocab_path = args.save_dir / VOCAB_FILENAME
    vocab.save(vocab_path)

    print(
        f'Train markups saved: {train_markups_path} '
        f'({hf.format_size(utils.get_filesize(train_markups_path))})'
    )
    print(
        f'Test markups saved: {test_markups_path} '
        f'({hf.format_size(utils.get_filesize(test_markups_path))})'
    )
    print(f'Vocab saved: {vocab_path} ' f'({hf.format_size(utils.get_filesize(vocab_path))})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i',
        '--data-path',
        type=Path,
        help='a path to a raw wiki segments TXT file',
        default='data/ruwiki-yo-segments.txt',
    )
    parser.add_argument(
        '-O',
        '--save-dir',
        type=Path,
        help='a directory to save the markup files (train, test) and a vocab',
        default='data',
    )
    parser.add_argument(
        '--strategy',
        type=SampleStrategy,
        choices=SampleStrategy,
        default=SampleStrategy.DOWNSAMPLE,
        help='a sampling strategy to use for balancing word counts in the output dataset',
    )
    parser.add_argument(
        '-x',
        '--x-limit',
        metavar='INT',
        type=int,
        default=X_LIMIT,
        help=(
            'a maximum multiplication factor between a major and a minor word occurence counts '
            'in the output dataset (only for upperlimit strategy)'
        ),
    )
    parser.add_argument(
        '--test-size',
        metavar='FLOAT',
        type=float,
        help='a test fraction (or an absolute number of samples)',
        default=TEST_FRACTION,
    )
    parser.add_argument(
        '--min-word-count',
        metavar='INT',
        type=int,
        default=MIN_COUNT,
        help='a minimum number of occurences for a single Е/Ё word to be added in the vocab',
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
        '--compress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='whether to compress the output markup files',
    )
    parser.add_argument(
        '--vocab-filename', default=VOCAB_FILENAME, help='a filename to use for vocab'
    )
    parser.add_argument(
        '--markups-filename', default=MARKUPS_FILENAME, help='a filename to use for markups file'
    )
    main(parser.parse_args())
