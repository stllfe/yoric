from typing import Set

import pytest

from src import utils


@pytest.mark.parametrize('text, expected', [
    (
        'Платье было тёмным (даже чёрным), но он все равно сказал: «Вы светитесь, как звёздочка!»',
        {'Вы светитесь, как звёздочка!', 'даже чёрным', 'Платье было тёмным , но он все равно сказал: '},
    ),
])
def test_extract_unique_yo_segments(text: str, expected: Set[str]):
    assert set(utils.extract_unique_yo_segments(text)) == expected


@pytest.mark.parametrize('text, expected', [
    (
        'В «Повести временных лет» (XII век) упоминается этноним «литва», полностью совпадающий с названием '
        'местности «Литва» и по смыслу (территория, где живёт литва), и по форме.== География ==Поверхность равнинная, со следами древнего оледенения.\n',

        'В "Повести временных лет" ( век) упоминается этноним "литва", полностью совпадающий с названием местности "Литва" и по смыслу '
        '(территория, где живёт литва), и по форме. Поверхность равнинная, со следами древнего оледенения. '
    ),
])
def test_normalize_wiki_text(text: str, expected: str):
    assert utils.normalize_wiki_text(text) == expected
