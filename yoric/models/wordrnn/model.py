"""Implementation of a simple word-level CNN & BiLSTM architectures."""

from __future__ import annotations

import warnings

from typing import Any, cast, TypedDict

import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from yoric.data import Vocab

from .config import UNK
from .data import Sample


Samples = list[Sample]


class StateDict(TypedDict, total=False):
    """Saved model state from which it can be restored."""

    hparams: dict[str, Any]
    weights: dict[str, torch.Tensor]


def encode_word(word: str, vocab: Vocab, unk_warn: bool = False) -> int:
    """Converts a word to an integer.

    Vocab is expected to contain <unk> token for unknown words!
    """

    try:
        return cast(int, vocab[word])
    except KeyError:
        if unk_warn:
            warnings.warn(f'Unknown word: {word}', UserWarning)
        return cast(int, vocab[UNK])


class WordBiLSTM(nn.Module):
    """A recurrent word-level `Ё` replacement prediction model."""

    def __init__(
        self,
        num_words: int,
        emb_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim)
        self.hparams = {
            'num_words': num_words,
            'emb_dim': emb_dim,
            'num_layers': num_layers,
            'dropout': dropout,
        }
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )

    def load_embeddings(self, embs: torch.Tensor) -> None:
        self.emb.weight.data.copy_(embs)

    @classmethod
    def from_state(cls, state: StateDict) -> WordBiLSTM:
        model = cls(**state['hparams'])
        model.load_state_dict(state['weights'], strict=True)
        return model

    def to_state(self) -> StateDict:
        return StateDict(hparams=self.hparams, weights=self.state_dict())

    def forward(
        self, context: torch.LongTensor, lengths: torch.LongTensor, indices: torch.LongTensor
    ) -> torch.FloatTensor:
        embeddings = self.emb(context)
        targets = embeddings[torch.arange(embeddings.size(0)), indices]

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, *_ = self.lstm(packed)
        output, *_ = pad_packed_sequence(output, batch_first=True)

        context = output.sum(1)  #  / lengths.view(-1, 1)

        # print(targets.shape)
        # print(context.shape)

        t = torch.nn.functional.normalize(targets, dim=1)
        c = torch.nn.functional.normalize(context, dim=1)

        # print(torch.norm(t, p=2, dim=1))
        # print(torch.norm(c, p=2, dim=1))

        x = t.unsqueeze_(1) @ c.unsqueeze_(2)

        # [-1, 1] to [0, 1] range
        x = (x + 1) / 2
        return x.squeeze()
