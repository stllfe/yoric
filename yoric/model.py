"""An abstract `Ё` restoration model inteface."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import NamedTuple

from yoric.data import YeYoDataset


class YoWordSubstring(NamedTuple):
    start: int
    end: int
    score: float


class YoModel(ABC):
    """Abstract class for yofication model."""

    def fit(self, X: YeYoDataset) -> None:  # noqa
        pass

    @abstractmethod
    def predict(self, data: list[str], verbose: bool = False) -> list[list[YoWordSubstring]]:
        pass
