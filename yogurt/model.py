"""An abstract `Ё` restoration model inteface."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from yogurt.data import YeYoDataset


YoWordSubstrings = list[tuple[int, int]]


class YoModel(ABC):
    """Abstract class for yofication model."""

    def fit(self, X: YeYoDataset) -> None:  # noqa
        pass

    @abstractmethod
    def predict(self, data: list[str], verbose: bool) -> list[YoWordSubstrings]:
        pass
