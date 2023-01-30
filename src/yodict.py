"""Dictionary-based word lookup and replacement.

References:
https://github.com/e2yo/eyo-kernel/blob/fd95b1db9bec67298a5e2c5fab4078765eceaf7c/lib/dictionary.js
"""

from __future__ import annotations
from functools import lru_cache

from pathlib import Path
import re
from typing import Union


from src import consts


class YoDict:
    """A dictionary-based yoficator."""

    def __init__(self) -> None:
        self._dict = {}

    def __getitem__(self, key: str) -> str:
        return self._dict[key]

    def __len__(self) -> int:
        return len(self._dict)

    def clear(self) -> None:
        """Clears the dictionary."""

        self._dict = {}

    @classmethod
    def load(cls, path: Union[str, Path]) -> YoDict:
        """Loads the dictionary from a text file."""

        obj = cls()
        with open(path, mode='r', encoding='utf-8') as file:
            for word in file.readlines():
                word = word.split('#')[0]  # ignore comments
                obj.add_word(word.strip())
        return obj

    def add_word(self, word: str) -> None:
        """Adds given word forms to the dictionary."""

        if word.find('(') > -1:
            base, *parts = re.split(r'[(|)]', word)
            for part in parts:
                self._add_word(base + part)
        else:
            self._add_word(word)

    def _add_word(self, word: str) -> None:
        """Puts a single final word form to the dictionary."""

        word = word.replace('_', '', 1)
        key = self._replace_yo(word)

        self._dict[key] = word

        add_capitalized = not self._is_capitalized(word) and not self._has_underscore(word)

        if add_capitalized:
            self._dict[key.capitalize()] = word.capitalize()

    @staticmethod
    def _has_underscore(word: str) -> bool:
        return word.find('_') == 0

    @staticmethod
    def _is_capitalized(word: str) -> bool:
        return re.match(r'^[А-ЯЁ]', word) is not None

    def _replace_yo(self, word: str) -> str:
        """Replaces `Ё` with `Е`."""

        return word.replace('Ё', 'Е').replace('ё', 'е')

    def remove_word(self, word: str) -> None:
        """Removes a given word from the dictionary."""

        key = self._replace_yo(word)

        del self._dict[key]

        if not self._is_capitalized(key):
            del self._dict[key.capitalize()]

    def yoficate(self, word: str) -> str:
        """Restores `Ё` in a word if the word exists in the dictionary."""

        return self._dict.get(word, word)


@lru_cache()
def get_safe() -> YoDict:
    """Loads a dictionary with safe replacements only."""

    return YoDict.load(consts.SAFE_DICT_PATH)


@lru_cache()
def get_not_safe() -> YoDict:
    """Loads a dictionary with non-safe replacements."""

    return YoDict.load(consts.NOT_SAFE_DICT_PATH)
