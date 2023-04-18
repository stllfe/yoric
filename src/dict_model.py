from src import yodict
from src.eyo import Eyo
from src.model import YoModel, YoWordSubstrings
from tqdm import tqdm
from typing import List


class DictModel(YoModel):
    """Model that preidcts substrings of yo words from a dict"""

    def __init__(self, safe_dict: bool = True) -> None:
        super().__init__()
        yo_dict = yodict.get_safe() if safe_dict else yodict.get_not_safe()
        self.model = Eyo(yo_dict)

    def _predict(self, text: str) -> YoWordSubstrings:
        """Predicts yo word substrings for a single string"""
        res = []
        for replacement in self.model.lint(text):
            start_word = replacement.position.index
            end_word = start_word + len(replacement.after)
            res.append((start_word, end_word))
        return res

    def predict(self, data: List[str], verbose: bool = False) -> List[YoWordSubstrings]:
        """Predicts yo word substrings"""

        if verbose:
            data = tqdm(data, desc="Predict")

        return [self._predict(text) for text in data]
