from src import yodict
from src.eyo import Eyo
from src.model import yoModel
from tqdm import tqdm
from typing import Tuple, List


class Baseline(yoModel):

    def __init__(self) -> None:
        super().__init__()
        
        yo_dict = yodict.get_safe()
        self.model = Eyo(yo_dict)
    
    def _predict(self, text: str) -> List[Tuple[int, int]]:
        """Fuctuinon for yoword substring predictions by single string"""
        res = []
        for replacement in self.model.lint(text):
            start_word = replacement.position.index
            end_word = start_word + len(replacement.after)
            res.append((start_word, end_word))
        return res

    def predict(self, data: List[List[str]], show_progress: bool= False) -> List[List[Tuple[int, int]]]:
        """Function for yoword substring predictions"""
        iter_data = data

        if show_progress:
            iter_data = tqdm(data, desc="Predcit")
            
        res = [self._predict(text) for text in iter_data]    
        return res