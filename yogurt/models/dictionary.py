"""A baseline dictionary-based `Ё` letter restoration model."""


from tqdm import tqdm

from yogurt import yodict
from yogurt.eyo import Eyo
from yogurt.eyo import Position
from yogurt.model import YoModel
from yogurt.model import YoWordSubstrings


class DictModel(YoModel):
    """Model that predicts substrings of yo words from a dictionary."""

    def __init__(self, safe: bool = True) -> None:
        super().__init__()
        yo_dict = yodict.get_safe() if safe else yodict.get_not_safe()
        self.model = Eyo(yo_dict)

    def _predict(self, text: str) -> YoWordSubstrings:
        """Predicts yo word substrings for a single string."""

        res = []
        for replacement in self.model.lint(text):
            assert isinstance(replacement.position, Position)  # fix: see replacement class
            start_word = replacement.position.index
            end_word = start_word + len(replacement.after)
            res.append((start_word, end_word))
        return res

    def predict(self, data: list[str], verbose: bool = False) -> list[YoWordSubstrings]:
        """Predicts yo word substrings."""

        if verbose:
            data = tqdm(data, desc='Predict')  # type: ignore

        return [self._predict(text) for text in data]
