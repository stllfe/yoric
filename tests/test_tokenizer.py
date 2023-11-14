import pytest

from yoric.models.wordrnn.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(lower=True)


def test_dirty_string(tokenizer: Tokenizer) -> None:
    string = 'Тут используем всякие  "странные" == символы, типа 123; 342!!!'
    expect = ['тут', 'используем', 'всякие', 'странные', 'символы', 'типа']
    assert tokenizer(string) == expect


def test_clean_string(tokenizer: Tokenizer) -> None:
    string = 'А тут вот всё окей'
    expect = ['а', 'тут', 'вот', 'всё', 'окей']
    assert tokenizer(string) == expect


def test_hyphen_string(tokenizer: Tokenizer) -> None:
    string = 'Мы все-таки смогли'
    expect = ['мы', 'все', 'таки', 'смогли']
    result = tokenizer(string)
    assert result == expect
