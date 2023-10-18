import pytest

from yoric import data
from yoric import evaluate
from yoric.model import YoWordSubstring
from yoric.utils import Substring


# pylint: disable=arguments-out-of-order


def test_substring_equal_on_start_end() -> None:
    s1 = Substring(1, 10)
    s2 = YoWordSubstring(1, 10, 0.99)

    assert evaluate.equal_substrings(s1, s2)
    assert evaluate.equal_substrings(s2, s1)


def test_substring_equal_same_class() -> None:
    s1 = Substring(1, 10)
    s2 = Substring(1, 10)

    assert evaluate.equal_substrings(s1, s2)
    assert evaluate.equal_substrings(s2, s1)

    s1 = YoWordSubstring(1, 10, 0.6)  # type: ignore
    s2 = YoWordSubstring(1, 10, 0.5)  # type: ignore

    assert evaluate.equal_substrings(s1, s2)
    assert evaluate.equal_substrings(s2, s1)

    s1 = (5, 8)  # type: ignore
    s2 = (5, 8)  # type: ignore

    assert evaluate.equal_substrings(s1, s2)
    assert evaluate.equal_substrings(s2, s1)

    s3 = (8, 5)

    assert not evaluate.equal_substrings(s2, s3)
    assert not evaluate.equal_substrings(s3, s2)


# pylint: enable=arguments-out-of-order


MARKUPS = [
    data.YeYoMarkup('вышел ежик из тумана...', [Substring(6, 10)], [1], [1]),
    data.YeYoMarkup('все могут короли!', [Substring(0, 3)], [0], [1]),
]
IDEAL_PREDICTIONS = ([YoWordSubstring(6, 10, 1)], [YoWordSubstring(0, 3, 1)])
SCORE_PREDICTIONS = ([YoWordSubstring(6, 10, 0.5)], [YoWordSubstring(0, 3, 0.5)])
EMPTY_PREDICTIONS: tuple[list, list] = (
    [],
    [],
)
TRUES_BINARY = [1, 1]
IDEAL_BINARY = [1, 1]
EMPTY_BINARY = [0, 0]
SCORE_BINARY = [1, 1]
SCORE_SCORES = [0.5, 0.5]


def test_unroll_predictions_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate.unroll_predictions(MARKUPS, [])


@pytest.mark.parametrize(
    'preds,xpreds,xscores',
    [
        (IDEAL_PREDICTIONS, IDEAL_BINARY, IDEAL_BINARY),
        (EMPTY_PREDICTIONS, EMPTY_BINARY, EMPTY_BINARY),
        (SCORE_PREDICTIONS, SCORE_BINARY, SCORE_SCORES),
    ],
)
def test_unroll_predictions_on_ideal_case(
    preds: list[list[YoWordSubstring]], xpreds: list[int], xscores: list[float]
) -> None:
    t, p, s = evaluate.unroll_predictions(MARKUPS, preds)
    assert t == TRUES_BINARY
    assert p == xpreds
    assert s == xscores
