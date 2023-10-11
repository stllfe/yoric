"""Project-level helpers and utility functions."""

from __future__ import annotations

import bz2
import io
import itertools
import json
import os
import re
import unicodedata

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, IO, NamedTuple, Optional, Union

import razdel
import simdjson
import yaml

from yoric.yodict import YoDict


EMPTY = ''
SPACE = ' '
SEPARATOR = '#'

YO_LOWER = 'ё'
YE_LOWER = 'е'

WIKI_HEADER = '=='

PUNCTUATION = '[-–—{}()[\\]|<>=\\_"\'«»„“#$^%&*+:;.,?!]'
WORDS_REGEX = re.compile(
    r'([А-ЯЁа-яё])[а-яё]+(?![а-яё]|\\.[ \u00A0\t]+([а-яё]|[А-ЯЁ]{2}|'
    + PUNCTUATION
    + ')|\\.'
    + PUNCTUATION
    + ')',
    re.MULTILINE,
)

ALLOWED_SYMBOLS = '[-А-яЁё0-9 .,?!()–—:;"«»]'
ALLOWED_SYMBOLS_REGEX = re.compile(ALLOWED_SYMBOLS)

NOT_ALLOWED_SYMBOLS = f'[^{ALLOWED_SYMBOLS[1:-1]}]'
NOT_ALLOWED_SYMBOLS_REGEX = re.compile(NOT_ALLOWED_SYMBOLS)

QUOTE_REGEX = re.compile('((?:"|«|„)([-А-яЁё0-9 .,?!()––:;]+)(?:"|»|“))')
PARENTHESES_REGEX = re.compile(r'(\(([\-А-яЁё0-9 .,?!––:;"«»]+)\))')

MULTIPLE_SPACES_REGEX = re.compile(r'\s{2,}')
WIKI_HEADER_REGEX = re.compile(rf'=={ALLOWED_SYMBOLS}+==')
GLUED_SENTENCES_BORDER_REGEX = re.compile(r'([\w\d\s])([.;!?])([А-ЯЁ])')
HANGING_PUNCT_REGEX = re.compile(r'\s+([.,:;!?])')

JSON_PARSER = simdjson.Parser()


class Substring(NamedTuple):
    start: int
    end: int


def save_jsonl(objs: Iterable[Any], filepath: Union[str, Path], compress: bool = False) -> None:
    filepath = Path(filepath)
    fp = io.BytesIO()
    for obj in objs:
        s = json.dumps(obj, ensure_ascii=False) + '\n'
        fp.write(s.encode('utf-8'))
    if compress:
        data = bz2.compress(fp.getbuffer())
    else:
        fp.seek(0)
        data = fp.getvalue()
    with open(filepath, mode='wb') as fout:
        fout.write(data)


def load_jsonl(filepath: Union[str, Path], decompress: bool = False) -> Iterable[dict]:
    filepath = Path(filepath)
    opener: Callable[[Union[str, Path]], IO[Any]] = open
    if '.bz2' in filepath.suffixes or decompress:
        opener = bz2.open
    with opener(filepath) as fp:
        for line in fp:
            parsed = JSON_PARSER.parse(line)
            try:
                yield parsed.as_dict()  # type: ignore
            finally:
                # to avoid C memory linking issues on error
                del parsed


def load_yaml(filepath: Union[str, Path], section: Optional[str] = None) -> Any:
    with open(filepath, encoding='utf-8') as fp:
        data = yaml.safe_load(fp)
        for subs in section.split('.') if section else ():
            data = data[subs]
        return data  # returns Any by design


def split_sentences(text: str) -> list[str]:
    """Splits texts onto sentences as accurately as possible."""

    split = []
    for sentence in razdel.sentenize(text):
        split.append(sentence.text.strip())
    return split


def normalize_quote_marks(text: str) -> str:
    """Converts all quote marks to `"`."""

    return QUOTE_REGEX.sub(r'"\2"', text)


def extract_unique_yo_segments(text: str, clean: bool = False, repl: str = EMPTY) -> list[str]:
    """Extracts all unique quotes, parentheses and whole sentences that contain `Ё`."""

    segments = []

    for sentence in split_sentences(text):
        if clean:
            quotes, sentence = extract_quotes(sentence, repl=repl, return_text=True)
            parentheses, sentence = extract_parentheses(sentence, repl=repl, return_text=True)
        else:
            # todo: remove this return stuff, better decompose so that MyPy is happy
            quotes = extract_quotes(sentence, return_text=False)  # type: ignore
            parentheses = extract_parentheses(sentence, return_text=False)  # type: ignore

        for item in quotes + parentheses + [sentence]:  # type: ignore
            if YO_LOWER in item.lower():
                segments.append(item)
    return segments


def remove_multiple_spaces(text: str, repl: str = SPACE) -> str:
    """Removes sequences of more than one space symbol."""

    return MULTIPLE_SPACES_REGEX.sub(repl, text)


def remove_not_allowed_symbols(text: str, repl: str = SPACE) -> str:
    """Removes all symbols that are not alphanumeric or punctuation."""

    return NOT_ALLOWED_SYMBOLS_REGEX.sub(repl, text)


def extract_quotes(
    text: str, repl: str = EMPTY, return_text: bool = False
) -> Union[list[str], tuple[list[str], str]]:
    """Extracts all quotes (without quote marks) from the text, optionally returning the cleaned
    text."""

    output = []
    for item in QUOTE_REGEX.finditer(text):
        outer, inner = item.groups()
        text = text.replace(outer, repl)
        output.append(inner)
    return (output, text) if return_text else output


def extract_parentheses(
    text: str, repl: str = EMPTY, return_text: bool = False
) -> Union[list[str], tuple[list[str], str]]:
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
    """Returns the file size in bytes."""

    return os.stat(filepath).st_size


def get_dirsize(dirpath: Union[str, Path]) -> int:
    """Returns the directory size in bytes."""

    dirpath = Path(dirpath)
    total = 0
    for p in dirpath.glob('*'):
        if p.is_file():
            total += get_filesize(p)
        elif p.is_dir():
            total += get_dirsize(p)
    return total


def hasyo(text: str) -> bool:
    """Returns whether or not the given text contains `Ё` letter."""

    return YO_LOWER in text.lower()


def hasye(text: str) -> bool:
    """Returns whether or not the given text contains `Е` letter."""

    return YE_LOWER in text.lower()


def hasyeyo(text: str) -> bool:
    """Returns whether or not the given text contains either `Е` or `Ё` letters."""

    return hasyo(text) or hasye(text)


def get_yo_substrings(text: str, yodict: Optional[YoDict] = None) -> list[Substring]:
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
        substrings.append(Substring(*match.span()))

    return substrings


def get_ye_substrings(text: str, yodict: Optional[YoDict] = None) -> list[Substring]:
    """Returns all `Е` substring tuples (start, end indices).

    Args:
        text: A text to extract the substrings from.
        yodict: A dictionary to use for selecting substrings (`None` = all substrings)

    """

    substrings = []
    for match in WORDS_REGEX.finditer(text):
        if not hasye(match.group()) or hasyo(match.group()):
            continue
        if yodict and match.group() not in yodict:
            continue
        substrings.append(Substring(*match.span()))
    return substrings


def yeficate(text: str) -> str:
    """Replaces `Ё` to `Е` letters."""

    return text.replace('Ё', 'Е').replace('ё', 'е')
