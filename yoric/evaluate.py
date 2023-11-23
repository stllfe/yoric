"""Model performance evaluation stuff."""

from __future__ import annotations

import time

from collections.abc import ItemsView
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from typing import Final, Literal, Protocol, Union

from sklearn import metrics as M
from tabulate import tabulate

from yoric import utils
from yoric.data import YeYoDataset
from yoric.data import YeYoMarkup
from yoric.model import YoModel
from yoric.model import YoWordSubstring


SubstringLike = Union[YoWordSubstring, utils.Substring, tuple[int, int]]
TableStyle = Literal[
    'plain', 'simple', 'grid', 'simple_grid', 'rounded_grid', 'outline', 'rounded_outline'
]

TABLE_DEFAULT_STYLE: Final = 'rounded_outline'
TABLE_NUMBER_FORMAT = '.9g'


class ItemsMixin(Protocol):
    def items(self) -> ItemsView[str, float]:
        return asdict(self).items()  # type: ignore


@dataclass(frozen=True)
class Timings(ItemsMixin):
    """Model speed measurements."""

    wall_time: float
    cpu_time: float
    tokens_per_sec: float


@dataclass(frozen=True)
class Metrics(ItemsMixin):
    """Model quality metrics."""

    accuracy: float
    precision: float
    recall: float
    fp: int
    fn: int
    support: int
    log_loss: float
    f1_score: float
    fh_score: float
    auroc: float

    @classmethod
    def from_predictions(cls, trues: list[int], preds: list[int], scores: list[float]) -> Metrics:
        """Calculates the metrics from the given inputs and outputs."""

        _, fp, fn, tp = M.confusion_matrix(trues, preds).ravel()

        return Metrics(
            accuracy=float(M.accuracy_score(trues, preds)),
            precision=float(M.precision_score(trues, preds)),
            recall=float(M.recall_score(trues, preds)),
            fp=int(fp),
            fn=int(fn),
            support=int(tp) + int(fn),
            log_loss=float(M.log_loss(trues, scores)),
            f1_score=float(M.f1_score(trues, preds)),
            fh_score=float(M.fbeta_score(trues, preds, beta=0.5)),
            auroc=float(M.roc_auc_score(trues, scores)),
        )


@dataclass
class Evaluation:
    """All-around model performance evaluation."""

    metrics: Metrics
    timings: Timings

    def table(self) -> str:
        return make_table({**asdict(self.metrics), **asdict(self.timings)})


def evaluate_model(model: YoModel, dataset: YeYoDataset, verbose: bool = False) -> Evaluation:
    """Runs metrics calculation on the given model and dataset."""

    X = [utils.yeficate(markup.text) for markup in dataset]

    wall_time = time.perf_counter()
    cpu_time = time.process_time()

    outs = model.predict(X, verbose=verbose)

    wall_time = time.perf_counter() - wall_time
    cpu_time = time.process_time() - cpu_time

    trues, preds, scores = unroll_predictions(list(dataset), outs)

    metrics = Metrics.from_predictions(trues, preds, scores)
    timings = Timings(wall_time, cpu_time, sum(map(utils.get_tokens_count, X)) / wall_time)

    return Evaluation(metrics, timings)


def unroll_predictions(
    markups: list[YeYoMarkup], preds: list[list[YoWordSubstring]]
) -> tuple[list[int], list[int], list[float]]:
    """Flattens and aligns model predictions with the true labels from the markups.

    Returns:
        A tuple of `true`, `pred`, `score` lists.
    """

    if len(markups) != len(preds):
        raise ValueError(
            f'Length mismatch between markups and predictions: {len(markups)} and {len(preds)}!'
        )

    trues_binary: list[int] = []
    preds_binary: list[int] = []
    preds_scores: list[float] = []

    for markup, pred in zip(markups, preds):
        trues_binary.extend(markup.targets)
        for s1 in markup.spans:
            for s2 in pred:
                if not equal_substrings(s1, s2):
                    continue
                preds_binary.append(1)
                preds_scores.append(s2.score)
                break
            else:
                preds_binary.append(0)
                preds_scores.append(0)

    return trues_binary, preds_binary, preds_scores


def make_table(
    mapping: ItemsMixin,
    precision: int = 3,
    headers: Sequence[str] = (),
    style: TableStyle = TABLE_DEFAULT_STYLE,
) -> str:
    """Formats evaluation result as a table."""

    return tabulate(
        [[k, round(v, precision)] for k, v in mapping.items()],
        floatfmt=TABLE_NUMBER_FORMAT,
        tablefmt=style,
        headers=headers,
    )


def equal_substrings(s1: SubstringLike, s2: SubstringLike) -> bool:
    """Compares two substring tuples by the first two elements (start, end)."""

    return s1[:2] == s2[:2]
