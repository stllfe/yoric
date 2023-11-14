"""Data related utilities."""

import random

from collections.abc import Iterator
from typing import Callable, NamedTuple

from yoric.data import YeYoDataset
from yoric.utils import yeficate


class Sample(NamedTuple):
    words: list[str]
    index: int
    target: int


def batchify(
    dataset: YeYoDataset,
    tokenizer: Callable[[str], list[str]],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[list[Sample]]:
    """Sequence classification batcher for :class:`YeYoDataset`."""

    markups = dataset.markups
    vocab = dataset.vocab

    if shuffle:
        random.shuffle(markups)

    items: list[Sample] = []
    for markup in dataset:
        words = tokenizer(yeficate(markup.text))
        for label, target in zip(markup.labels, markup.targets):
            word: str = vocab[label]
            word = word.lower()
            try:
                index = words.index(yeficate(word))
            except (ValueError, IndexError):
                # print(f'Bad sample after tokenization: {markup}')
                continue
            items.append(Sample(words, index, target))
            if len(items) == batch_size:
                yield items
                items.clear()
    if 0 < len(items) < batch_size and not drop_last:
        yield items


def count_batches(dataset: YeYoDataset, batch_size: int, drop_last: bool = False) -> int:
    total = sum(map(lambda m: len(m.labels), dataset))
    count, remainder = divmod(total, batch_size)
    return count + (0 if drop_last else bool(remainder))
