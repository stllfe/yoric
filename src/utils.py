import itertools
import os
import re
import unicodedata

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import razdel

from src.yodict import YoDict
from src.yodict import get_not_safe


EMPTY = ''
SPACE = ' '
SEPARATOR = '#'

YO_LOWER_SYMBOL = 'ё'
WIKI_HEADER = '=='

PUNCTUATION = '[{}()[\\]|<>=\\_"\'«»„“#$^%&*+-:;.,?!]'
WORDS_REGEX = re.compile(
    r'([А-ЯЁа-яё])[а-яё]+(?![а-яё]|\\.[ \u00A0\t]+([а-яё]|[А-ЯЁ]{2}|' +
    PUNCTUATION + ')|\\.' +
    PUNCTUATION + ')',
    re.MULTILINE
)

ALLOWED_SYMBOLS = r'[А-яЁё0-9 .,?!()\-––:;"«»]'
ALLOWED_SYMBOLS_REGEX = re.compile(ALLOWED_SYMBOLS)

NOT_ALLOWED_SYMBOLS = rf'[^{ALLOWED_SYMBOLS[1:-1]}]'
NOT_ALLOWED_SYMBOLS_REGEX = re.compile(NOT_ALLOWED_SYMBOLS)

QUOTE_REGEX = re.compile(r'((?:"|«|„)([А-яЁё0-9 .,?!()\-––:;]+)(?:"|»|“))')
PARENTHESES_REGEX = re.compile(r'(\(([А-яЁё0-9 .,?!\-––:;"«»]+)\))')

MULTIPLE_SPACES_REGEX = re.compile(r'\s{2,}')
WIKI_HEADER_REGEX = re.compile(rf'=={ALLOWED_SYMBOLS}+==')
GLUED_SENTENCES_BORDER_REGEX = re.compile(r'([\w\d\s])([.;!?])([А-ЯЁ])')
HANGING_PUNCT_REGEX = re.compile(r'\s+([.,:;!?])')

Substring = Tuple[int, int]


def split_sentences(text: str) -> List[str]:
    """Splits texts onto sentences as accurately as possible."""

    split = []
    for sentence in razdel.sentenize(text):
        split.append(sentence.text.strip())
    return split


def normalize_quote_marks(text: str) -> str:
    """Converts all quote marks to `"`."""

    return QUOTE_REGEX.sub(r'"\2"', text)


def extract_unique_yo_segments(text: str, clean: bool = False, repl: str = EMPTY) -> List[str]:
    """Extracts all unique quotes, parentheses and whole sentences that contain `Ё`."""

    segments = []

    for sentence in split_sentences(text):
        if clean:
            quotes, sentence = extract_quotes(sentence, repl=repl, return_text=True)
            parentheses, sentence = extract_parentheses(sentence, repl=repl, return_text=True)
        else:
            quotes = extract_quotes(sentence, return_text=False)
            parentheses = extract_parentheses(sentence, return_text=False)

        for item in quotes + parentheses + [sentence]:
            if YO_LOWER_SYMBOL in item.lower():
                segments.append(item)
    return segments


def remove_multiple_spaces(text: str, repl: str = SPACE) -> str:
    """Removes sequences of more than one space symbol."""

    return MULTIPLE_SPACES_REGEX.sub(repl, text)


def remove_not_allowed_symbols(text: str, repl: str = SPACE) -> str:
    """Removes all symbols that are not alphanumeric or punctuation."""

    return NOT_ALLOWED_SYMBOLS_REGEX.sub(repl, text)


def extract_quotes(text: str, repl: str = EMPTY, return_text=False) -> Union[List[str], Tuple[List[str], str]]:
    """Extracts all quotes (without quote marks) from the text, optionally returning the cleaned text."""

    output = []
    for item in QUOTE_REGEX.finditer(text):
        outer, inner = item.groups()
        text = text.replace(outer, repl)
        output.append(inner)
    return (output, text) if return_text else output


def extract_parentheses(text: str, repl: str = EMPTY, return_text=False) -> Union[List[str], Tuple[List[str], str]]:
    """Extracts all parentheses from the text, returning the text without them (optionally)."""

    output = []
    for item in PARENTHESES_REGEX.finditer(text):
        outer, inner = item.groups()
        text = text.replace(outer, repl)
        output.append(inner)
    return (output, text) if return_text else output


def normalize_unicode(text: str) -> str:
    """Converts ambiguos unicode symbols to more simple and common ones."""

    return unicodedata.normalize('NFKC', text)


def remove_newlines(text: str, repl: str = SPACE) -> str:
    """Removes newlines from the text."""

    return text.replace('\n', repl)


def remove_wiki_header(text: str, repl: str = SPACE) -> str:
    """Removes wiki headers from the text."""

    return WIKI_HEADER_REGEX.sub(repl, text)


def restore_glued_sentences(text: str, sep: str = SPACE) -> str:
    """Naively restores the space between two (likely) sentences."""

    return GLUED_SENTENCES_BORDER_REGEX.sub(rf'\1\2{sep}\3', text)


def fix_hanging_punctuation(text: str) -> str:
    """Removes excessive space symbols before the punctuation ones."""

    return HANGING_PUNCT_REGEX.sub(r'\1', text)


def normalize_wiki_text(text: str) -> str:
    """Produces as clean wiki text as possible."""

    x = normalize_unicode(text)
    x = remove_wiki_header(x)
    x = remove_not_allowed_symbols(x)
    x = remove_multiple_spaces(x)
    x = remove_newlines(x)
    x = restore_glued_sentences(x)
    x = fix_hanging_punctuation(x)

    return x


def batch(iterable: Iterable, size: int) -> Iterable:
    """Iterates over the given iterable in batches of fixed size."""

    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item


def get_filesize(filepath: Union[str, Path]) -> int:
    """Returns filesize in bytes."""

    return os.stat(filepath).st_size


def hasyo(text: str) -> bool:
    """Returns whether or not the given text contains `Ё` letter."""

    return YO_LOWER_SYMBOL in text.lower()


def get_yo_substrings(text: str, yodict: Optional[YoDict] = None) -> List[Substring]:
    """Returns all `Ё` substring tuples (start, end indices).

    Args:
        text: A text to extract the substrings from.
        yodict: A dictionary to use for selecting substrings (`None` = all substrings)
    """

    substrings = []
    for match in WORDS_REGEX.finditer(text):
        if not hasyo(match.group()):
            continue
        if yodict and match.group() not in yodict:
            continue
        substrings.append(match.span())

    return substrings


def get_not_safe_yo_substrings(text: str) -> List[Substring]:
    """Just a simple alias around `get_yo_substrings` and a not-safe dictionary."""

    return get_yo_substrings(text, yodict=get_not_safe())


def yeficate(text: str) -> str:
    """Replaces `Ё` to `Е` letters."""

    return text.replace('Ё', 'Е').replace('ё', 'е')
