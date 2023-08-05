import pytest

from yoric import yodict


@pytest.mark.parametrize(
    'word, forms, capitalized',
    [
        ('узна(ёмся|ётесь|ёшься)', ['узнаёмся', 'узнаётесь', 'узнаёшься'], True),
        (
            'ёжик(|а|ам|ами|ах|е|и|ов|ом|у)',
            [
                'ёжик',
                'ёжика',
                'ёжикам',
                'ёжиками',
                'ёжиках',
                'ёжике',
                'ёжики',
                'ёжиков',
                'ёжиком',
                'ёжику',
            ],
            True,
        ),
    ],
)
def test_add_word_adds_all_forms(word: str, forms: list[str], capitalized: bool) -> None:
    yd = yodict.YoDict()
    yd.add_word(word)

    assert len(yd) == len(forms) + (0, len(forms))[capitalized]

    for form in forms:
        assert form in yd


def test_yodict_is_iterable() -> None:
    yd = yodict.YoDict()
    yd.add_word('кент(|ик|у|ы)')
    words = ('кент', 'кентик', 'кенту', 'кенты')
    assert len(yd) == len(words) * 2

    idx = 0
    for idx, word in enumerate(yd, start=1):
        assert word.lower() in words

    assert idx == len(words) * 2
