"""A baseline dictionary-based `Ё` letter restoration model."""


from tqdm import tqdm

from yoric import yodict
from yoric.eyo import Eyo
from yoric.eyo import Position
from yoric.model import YoModel
from yoric.model import YoWordSubstring


class DictModel(YoModel):
    """Model that predicts substrings of yo words from a dictionary."""

    def __init__(self, safe: bool = True) -> None:
        super().__init__()
        self.model = Eyo(yodict.get_safe() if safe else yodict.get_not_safe())

    def _predict(self, text: str) -> list[YoWordSubstring]:
        """Predicts yo word substrings for a single string."""

        spans = []
        for replacement in self.model.lint(text):
            assert isinstance(replacement.position, Position)  # fix: see replacement class
            start = replacement.position.index
            end = start + len(replacement.after)
            spans.append(YoWordSubstring(start, end, 1))
        return spans

    def predict(self, data: list[str], verbose: bool = False) -> list[list[YoWordSubstring]]:
        """Predicts yo word substrings."""

        if verbose:
            data = tqdm(data, desc='Predict')  # type: ignore

        return [self._predict(text) for text in data]
