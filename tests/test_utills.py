from typing import Set

import pytest

from src import utils


@pytest.mark.parametrize('text, expected', [
    (
        'Платье было тёмным (даже чёрным), но он все равно сказал: «Вы светитесь, как звёздочка!»',
        {'Вы светитесь, как звёздочка!', 'даже чёрным', 'Платье было тёмным , но он все равно сказал: '},
    ),
])
def test_extract_unique_yo_segments(text: str, expected: Set[str]) -> None:
    assert set(utils.extract_unique_yo_segments(text, clean=True)) == expected


@pytest.mark.parametrize('text, expected', [
    (
        'В «Повести временных лет» (XII век) упоминается этноним «литва», полностью совпадающий с названием '
        'местности «Литва» и по смыслу (территория, где живёт литва), и по форме.== География ==Поверхность равнинная, со следами древнего оледенения.\n',

        'В «Повести временных лет» ( век) упоминается этноним «литва», полностью совпадающий с названием местности «Литва» и по смыслу '
        '(территория, где живёт литва), и по форме. Поверхность равнинная, со следами древнего оледенения. '
    ),
])
def test_normalize_wiki_text(text: str, expected: str) -> None:
    assert utils.normalize_wiki_text(text) == expected


@pytest.mark.parametrize('text, expected', [
    ('Жили-были дед , бабка, кошка да собака .', 'Жили-были дед, бабка, кошка да собака.'),
    ('Не странно ли это   ?', 'Не странно ли это?'),
    ('А вот тут все, в общем-то, ок!', 'А вот тут все, в общем-то, ок!'),
    ('   ', '   '),
    ('1 + 1', '1 + 1'),
    ('И он сказал: "Меня не трогать!"', 'И он сказал: "Меня не трогать!"'),
    ('И тут — лучше не стоит!', 'И тут — лучше не стоит!'),
    ('А вы кто ? !', 'А вы кто?!')
])
def test_fix_hanging_punctuation(text: str, expected: str) -> None:
    assert utils.fix_hanging_punctuation(text) == expected


@pytest.mark.parametrize('text, sep, expected', [
    ('Вот первое предложение.И сразу второе!', ' ', 'Вот первое предложение. И сразу второе!'),
    ('Вот первое предожение, Иван, не обессудьте.', ' ', 'Вот первое предожение, Иван, не обессудьте.'),
    ('Раз сказал!Два сказал', '\n', 'Раз сказал!\nДва сказал')
])
def test_restore_glued_sentences(text: str, sep: str, expected: str) -> None:
    assert utils.restore_glued_sentences(text, sep) == expected
