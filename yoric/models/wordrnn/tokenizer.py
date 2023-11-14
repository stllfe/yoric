"""Text preprocessing and tokenization logic."""

import re
import string

from collections.abc import Iterable

import razdel

from razdel.substring import Substring


LETTERS = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'

PUNCTS = string.punctuation
SPACES = string.whitespace
DIGITS = string.digits

HYPHEN = '-'
SPACE = ' '
DASH = '—'


class Tokenizer:
    # todo: looks like the vocab can be merged into this class
    # so that we would mimic HuggingFace tokenizers and maybe even interchange them later
    # if we need to
    """Splits texts into words."""

    def __init__(self, lower: bool = True, allow: str = LETTERS) -> None:
        self._lower = lower
        self._regex = re.compile(rf'[{allow}]')

    def normalize(self, text: str) -> str:
        return text.lower() if self._lower else text

    def tokenize(self, text: str) -> Iterable[Substring]:
        text = self.normalize(text)
        for token in tokenize_with_hyphen(text):
            if self._regex.match(token.text):
                yield token

    def __call__(self, text: str) -> list[str]:
        return [token.text for token in self.tokenize(text)]


def tokenize_with_hyphen(text: str) -> Iterable[Substring]:
    """Tokenizes words with hyphen as separate words."""

    # todo: may be simplified by extending/modifying razdel's logic?
    # consider reutilizing the word regex from utils?
    for token in razdel.tokenize(text):
        if HYPHEN in token.text:
            offset = token.text.index(HYPHEN)
            stop = token.start + offset
            yield Substring(token.start, stop, token.text[:offset])
            yield Substring(stop, stop + 1, HYPHEN)
            token.start = stop + 1
            token.text = token.text[offset + 1 :]
            yield token
            continue
        yield token
