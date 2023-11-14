"""API for the WordRNN model."""

from __future__ import annotations

from typing import Optional, Union

import torch

from tqdm import tqdm

from yoric import consts
from yoric import utils
from yoric import yodict
from yoric.data import Vocab
from yoric.model import YoModel
from yoric.model import YoWordSubstring

from . import config
from .model import encode_word
from .model import StateDict
from .model import WordBiLSTM
from .tokenizer import Tokenizer


class WordRNNModel(YoModel):
    """High-level API for WordRNN model."""

    def __init__(
        self,
        *,
        model: Optional[WordBiLSTM] = None,
        vocab: Optional[Vocab] = None,
        threshold: float = 0.5,
        device: Union[torch.device, str] = consts.CPU,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.device = device

        self.model = model or WordBiLSTM.from_state(
            StateDict(torch.load(config.MODEL_PATH, map_location=device))
        )
        self.model.eval()
        self.model.to(device)

        self.vocab = vocab or Vocab.load(config.VOCAB_PATH)
        self.tokenizer = Tokenizer(lower=True)

        self.not_safe_dict = yodict.get_not_safe()
        self.safe_dict = yodict.get_safe()

    @torch.no_grad()
    def _predict(self, text: str) -> list[YoWordSubstring]:
        words = list(self.tokenizer.tokenize(text))
        codes = [encode_word(word.text, self.vocab) for word in words]

        indices: list[int] = []
        lengths: list[int] = []
        context: list[list[int]] = []

        output: list[YoWordSubstring] = []

        def should_predict(word: str) -> bool:
            return (
                not utils.hasyo(word)
                and utils.hasye(word)
                and word in self.not_safe_dict
                and word in self.vocab
            )

        # todo: I think we should move this to some higher level API
        # like it's not this particular model's duty to handle dictionary words
        for index, word in enumerate(words):
            if word.text in self.safe_dict:
                output.append(YoWordSubstring(word.start, word.stop, 1))
                continue
            if should_predict(word.text):
                indices.append(index)
                context.append(codes)
                lengths.append(len(codes))
        if not indices:
            return output
        with torch.device(self.device):
            outs = self.model.forward(
                context=torch.tensor(context).view(len(indices), -1),
                lengths=torch.tensor(lengths),
                indices=torch.tensor(indices),
            )
        outs: torch.Tensor = torch.atleast_1d(outs)  # .sigmoid_())

        for index, score, check in zip(indices, outs, outs > self.threshold):
            word = words[index]
            if check:
                output.append(YoWordSubstring(word.start, word.stop, score.item()))
        return output

    def predict(self, data: list[str], verbose: bool = False) -> list[list[YoWordSubstring]]:
        if verbose:
            data = tqdm(data, desc='Predict')  # type: ignore

        return [self._predict(text) for text in data]
