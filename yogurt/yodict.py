"""Dictionary-based word lookup and replacement.

References:
https://github.com/e2yo/eyo-kernel/blob/fd95b1db9bec67298a5e2c5fab4078765eceaf7c/lib/dictionary.js

"""

from __future__ import annotations

import re

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Union

from yogurt import consts


class YoDict:
    """A dictionary-based yoficator."""

    def __init__(self) -> None:
        self._dict: dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        return self._dict[key]

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, key: str) -> bool:
        ekey = self._replace_yo(key)
        return self._dict.get(key, self._dict.get(ekey)) is not None

    def __iter__(self) -> Iterator[str]:
        all_words = set(self._dict.keys()) | set(self._dict.values())
        return iter(all_words)

    def clear(self) -> None:
        """Clears the dictionary."""

        self._dict = {}

    @classmethod
    def load(cls, path: Union[str, Path]) -> YoDict:
        """Loads the dictionary from a text file."""

        obj = cls()
        with open(path, encoding='utf-8') as file:
            for word in file.readlines():
                word = word.split('#')[0]  # ignore comments
                obj.add_word(word.strip())
        return obj

    def add_word(self, word: str) -> None:
        """Adds given word forms to the dictionary."""

        if '(' in word:
            base, parts = filter(bool, re.split(r'\(|\)', word))
            for part in parts.split('|'):
                self._add_word(base + part)
        else:
            self._add_word(word)

    def _add_word(self, word: str) -> None:
        """Puts a single final word form to the dictionary."""

        has_underscore = word.startswith('_')
        word = word.lstrip('_')

        is_capitalised = word[0].isupper()

        key = self._replace_yo(word)

        self._dict[key] = word

        add_capitalized = not is_capitalised and not has_underscore

        if add_capitalized:
            self._dict[key.capitalize()] = word.capitalize()

    def _replace_yo(self, word: str) -> str:
        """Replaces `Ё` with `Е`."""

        return word.replace('Ё', 'Е').replace('ё', 'е')

    def remove_word(self, word: str) -> None:
        """Removes a given word from the dictionary."""

        key = self._replace_yo(word)
        del self._dict[key]

        if not key[0].isupper():
            del self._dict[key.capitalize()]

    def restore_word(self, word: str) -> str:
        """Restores `Ё` in a word if the word exists in the dictionary."""

        return self._dict.get(word, word)


@lru_cache
def get_safe() -> YoDict:
    """Loads a dictionary with safe replacements only."""

    return YoDict.load(consts.SAFE_DICT_PATH)


@lru_cache
def get_not_safe() -> YoDict:
    """Loads a dictionary with non-safe replacements."""

    return YoDict.load(consts.NOT_SAFE_DICT_PATH)
