"""Dictionary-based word lookup and replacement.

References:
https://github.com/e2yo/eyo-kernel/blob/09595c4b254027d92ac0776ea9c60a74e26be733/lib/eyo.js
"""

import re

from dataclasses import dataclass
from functools import cmp_to_key
from typing import List, Union

from src.yodict import YoDict


PUNCTUATION = '[{}()[\\]|<>=\\_"\'«»„“#$^%&*+-:;.,?!]'
WORDS_REGEX = re.compile(
    r'([А-ЯЁа-яё])[а-яё]+(?![а-яё]|\\.[ \u00A0\t]+([а-яё]|[А-ЯЁ]{2}|' +
    PUNCTUATION + ')|\\.' +
    PUNCTUATION + ')',
    re.MULTILINE
)


@dataclass(frozen=True)
class Position:
    """A single word position in the given text."""

    line: int
    column: int
    index: int


@dataclass
class Replacement:
    """A single word replacement."""

    before: str
    after: str
    position: Union[Position, List[Position]]

    @property
    def count(self) -> int:
        """How many replacements of the given word were made."""

        return len(self.position) if not isinstance(self.position, Position) else 1


class Eyo:
    """The main class for text yofication."""

    def __init__(self, dictionary: YoDict) -> None:
        self.dictionary = dictionary

    def lint(self, text: str, group: bool = False) -> List[Replacement]:
        """Returns all the possible `Е` -> `Ё` replacements in the given text."""

        replacements = []

        if not text or not self._has_eyo(text):
            return replacements

        def replace(eword: re.Match[str]) -> str:
            pos = eword.start()
            eword = eword.group()
            yoword = self.dictionary.restore_word(eword)
            if yoword != eword:
                replacements.append(
                    Replacement(
                        before=eword,
                        after=yoword,
                        position=self._get_position(text, pos)
                    )
                )
                return yoword
            return eword

        text = re.sub(WORDS_REGEX, replace, text)
        if group:
            replacements = sorted(replacements, key=cmp_to_key(self._compare_replacements))
            replacements = self._remove_duplicates(replacements)

        return replacements

    def restore(self, text: str) -> str:
        """Restores `Ё` in the given text according to the dictionary."""

        if not text or not self._has_eyo(text):
            return text or ''

        def replace(eword: re.Match[str]) -> str:
            return self.dictionary.restore_word(eword.group())

        return re.sub(WORDS_REGEX, replace, text)

    @staticmethod
    def _has_eyo(text: str) -> bool:
        return re.search(r'[ЕЁеё]', text) is not None

    @staticmethod
    def _get_position(text: str, index: int) -> Position:
        lines = re.split(r'\r?\n', text[0:index])
        return Position(line=len(lines) - 1, column=len(lines[-1]), index=index)

    @staticmethod
    def _remove_duplicates(replacements: List[Replacement]) -> List[Replacement]:
        position = {}
        result = []
        for repl in replacements:
            position.setdefault(repl.before, []).append(repl.position)
        seen = set()
        for repl in replacements:
            if repl.before not in seen:
                repl.position = position[repl.before]
                result.append(repl)
                seen.add(repl.before)
        return result

    @staticmethod
    def _compare_replacements(left: Replacement, right: Replacement) -> int:
        l_before_lower = left.before.lower()
        r_before_lower = right.before.lower()

        if (left.before[0] != right.before[0]) and (l_before_lower[0] == r_before_lower[0]):
            return 1 if left.before > right.before else -1

        if l_before_lower > r_before_lower:
            return 1
        if l_before_lower < r_before_lower:
            return -1
        return 0
