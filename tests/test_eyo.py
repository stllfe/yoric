import pytest

from src import yodict
from src import eyo


@pytest.fixture()
def safe_eyo() -> eyo.Eyo:
    return eyo.Eyo(yodict.get_safe())


@pytest.fixture()
def not_safe_eyo() -> eyo.Eyo:
    return eyo.Eyo(yodict.get_not_safe())


@pytest.mark.parametrize('text,expected', [
    ('Вышел ежик на крыльцо!', eyo.Replacement(
        before='ежик',
        after='ёжик',
        position=eyo.Position(line=0, column=6, index=6),
    )),
])
def test_single_position(safe_eyo: eyo.Eyo, text: str, expected: eyo.Replacement):
    result = safe_eyo.lint(text)
    assert result == [expected]
    assert safe_eyo.restore(text) == text.replace(expected.before, expected.after)


# todo: copy other tests from the original repo
