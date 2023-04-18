from abc import ABC, abstractmethod
from typing import List, Tuple

YoWordSubstrings = List[Tuple[int, int]]


class YoModel(ABC):
    """Abstract class for yofication model"""

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, data: List[str], verbose: bool) -> List[YoWordSubstrings]:
        pass
