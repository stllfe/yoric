import itertools
import re

from typing import Iterable, List, Tuple, Union
import unicodedata

import razdel


EMPTY = ''
SPACE = ' '

YO_LOWER_SYMBOL = 'ё'
WIKI_HEADER = '=='

ALLOWED_SYMBOLS = r"[А-яЁё0-9\ \.,?!\(\)\-\–:;\"«»]"
ALLOWED_SYMBOLS_REGEX = re.compile(ALLOWED_SYMBOLS)

NOT_ALLOWED_SYMBOLS = f'[^{ALLOWED_SYMBOLS[1:-1]}]'
NOT_ALLOWED_SYMBOLS_REGEX = re.compile(NOT_ALLOWED_SYMBOLS)

QUOTE_REGEX = re.compile(r'((?:\"|«)([А-яЁё0-9\ \.,?!\(\)\-\–]+)(?:\"|»))')
PARENTHESES_REGEX = re.compile(r'(\(([А-яЁё0-9\ \.,?!\-\–\"«»]+)\))')

MULTIPLE_SPACES_REGEX = re.compile('\s{2,}')
WIKI_HEADER_REGEX = re.compile(f'=={ALLOWED_SYMBOLS}+==')


def split_sentences(text: str) -> List[str]:
    """Splits texts onto sentences as accurately as possible."""

    split = []
    for sentence in razdel.sentenize(text):
        split.append(sentence.text.strip())
    return split


def normalize_quote_marks(text: str) -> str:
    """Converts all quote marks to `"`."""

    return QUOTE_REGEX.sub(r'"\2"', text)


def extract_unique_yo_segments(text: str, repl: str = EMPTY) -> List[str]:
    """Extracts all unique quotes, parentheses and whole sentences that contain `Ё`."""

    sentences = []

    for sentence in split_sentences(text):
        quotes, sentence = extract_quotes(sentence, repl=repl, return_text=True)
        parentheses, sentence = extract_parentheses(sentence, repl=repl, return_text=True)
        for item in quotes + parentheses + [sentence]:
            if YO_LOWER_SYMBOL in item.lower():
                sentences.append(item)
    return sentences


def remove_multiple_spaces(text: str, repl: str = SPACE) -> str:
    """Removes sequences of more than one space symbol."""

    return MULTIPLE_SPACES_REGEX.sub(repl, text)


def remove_non_alphanumpunct(text: str, repl: str = SPACE) -> str:
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


def normalize_wiki_text(text: str) -> str:
    """Produces as clean wiki text as possible."""

    x = normalize_unicode(text)
    x = remove_wiki_header(x)
    x = remove_non_alphanumpunct(x)
    x = normalize_quote_marks(x)
    x = remove_newlines(x)
    x = remove_multiple_spaces(x)
    return x


def batch(iterable: Iterable, size: int) -> Iterable:
    """Iterates over the given iterable in batches of fixed size."""

    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item
