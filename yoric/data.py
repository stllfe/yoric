"""Data processing utils and helpers."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload, Union

from yoric import utils


class Vocab:
    """A simple vocab file for label encoding and word lookup."""

    def __init__(self, words: Iterable[str]) -> None:
        self._word2label = {}
        self._label2word = {}
        for index, word in enumerate(sorted(set(words))):
            self._label2word[index] = word
            self._word2label[word] = index

    def __len__(self) -> int:
        return len(self._word2label)

    def __iter__(self) -> Iterator[str]:
        return iter(self._label2word[i] for i in range(len(self)))

    def __contains__(self, key: Any) -> bool:
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __getitem__(self, key: Any) -> Union[str, int]:
        if isinstance(key, int):
            return self.get_word(key)
        if isinstance(key, str):
            return self.get_label(key)
        raise TypeError(f"Can't lookup keys of type {type(key)} in Vocab!")

    def get_label(self, word: str) -> int:
        """Returns a label by the given word."""

        return self._word2label[word]

    def get_word(self, label: int) -> str:
        """Returns a word by the given label."""

        return self._label2word[label]

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Vocab:
        """Loads vocab from a file."""

        words = []
        with open(filepath, encoding='utf-8') as vocab:
            for line in vocab:
                if word := line.strip():
                    words.append(word)
        return Vocab(words)

    def save(self, filepath: Union[str, Path]) -> None:
        """Dumps the given vocab to a file."""

        labels = sorted(self._label2word)
        with open(filepath, encoding='utf-8', mode='w') as vocab:
            for index in labels:
                vocab.write(self._label2word[index] + '\n')


@dataclass(frozen=True)
class YeYoMarkup:
    """A single dataset sample entity."""

    text: str
    spans: list[utils.Substring]
    labels: list[int]
    targets: list[int]


class YeYoDataset:
    """A simple iterator object for ye/yo markups."""

    def __init__(self, markups: Iterable[YeYoMarkup], vocab: Vocab) -> None:
        self._markups = list(markups)
        self._vocab = vocab

    @overload
    def __getitem__(self, index: int) -> YeYoMarkup:
        ...

    @overload
    def __getitem__(self, index: slice) -> YeYoDataset:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[YeYoMarkup, YeYoDataset]:
        if isinstance(index, int):
            return self.markups[index]
        if isinstance(index, slice):
            return YeYoDataset(self.markups[index], self.vocab)
        raise TypeError(f"Unsupported index type '{type(index)}'")

    def __len__(self) -> int:
        return len(self._markups)

    def __iter__(self) -> Iterator[YeYoMarkup]:
        return iter(self._markups)

    @property
    def vocab(self) -> Vocab:
        return self._vocab

    @property
    def markups(self) -> list[YeYoMarkup]:
        return self._markups


def save_markups(
    markups: Iterable[YeYoMarkup], filepath: Union[str, Path], compress: bool = False
) -> None:
    """Stores markups to a JSONL file with optional bzip compression."""

    return utils.save_jsonl(map(asdict, markups), filepath, compress=compress)


def load_markups(
    filepath: Union[str, Path],
    decompress: bool = False,
) -> Iterable[YeYoMarkup]:
    """Loads markups from a JSONL file with optional bzip compression."""

    for kwargs in utils.load_jsonl(filepath, decompress=decompress):
        kwargs['spans'] = [utils.Substring(*span) for span in kwargs['spans']]
        yield YeYoMarkup(**kwargs)


def load_dataset(
    markups_path: Union[str, Path],
    vocab_path: Union[str, Path],
    decompress: bool = False,
) -> YeYoDataset:
    """Loads whole markups dataset from disk."""

    return YeYoDataset(
        markups=load_markups(markups_path, decompress=decompress), vocab=Vocab.load(vocab_path)
    )
