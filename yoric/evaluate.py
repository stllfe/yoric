"""Model performance evaluation stuff."""

import re
import time

from collections.abc import ItemsView
from dataclasses import asdict
from dataclasses import dataclass
from typing import Literal, Union

from sklearn import metrics as M
from tabulate import tabulate

from yoric import utils
from yoric.data import YeYoDataset
from yoric.data import YeYoMarkup
from yoric.model import YoModel
from yoric.model import YoWordSubstring


SubstringLike = Union[YoWordSubstring, utils.Substring, tuple[int, int]]
TableFormat = Literal['plain', 'simple', 'grid', 'simple_grid', 'rounded_grid', 'outline']


@dataclass(frozen=True)
class Metrics:
    """Model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    log_loss: float
    f1_score: float
    fh_score: float
    auroc: float
    wall_time: float
    cpu_time: float
    tokens_per_sec: float

    def items(self) -> ItemsView[str, float]:
        return asdict(self).items()


def get_tokens_count(text: str) -> int:
    """Calculates count of tokens in a text."""

    return len(list(re.finditer(utils.WORDS_REGEX, text)))


def evaluate_model(model: YoModel, dataset: YeYoDataset, verbose: bool = False) -> Metrics:
    """Runs metrics calculation on the given model and dataset."""

    X = [utils.yeficate(markup.text) for markup in dataset]

    wall_time = time.perf_counter()
    cpu_time = time.process_time()

    outs = model.predict(X, verbose=verbose)

    wall_time = time.perf_counter() - wall_time
    cpu_time = time.process_time() - cpu_time

    trues, preds, scores = unroll_predictions(list(dataset), outs)

    metrics = Metrics(
        accuracy=float(M.accuracy_score(trues, preds)),
        precision=float(M.precision_score(trues, preds)),
        recall=float(M.recall_score(trues, preds)),
        log_loss=float(M.log_loss(trues, scores)),
        f1_score=float(M.f1_score(trues, preds)),
        fh_score=float(M.fbeta_score(trues, preds, beta=0.5)),
        auroc=float(M.roc_auc_score(trues, scores)),
        wall_time=wall_time,
        cpu_time=cpu_time,
        tokens_per_sec=sum(map(get_tokens_count, X)) / wall_time,
    )

    if verbose:
        print(make_table(metrics))

    return metrics


def unroll_predictions(
    markups: list[YeYoMarkup], preds: list[list[YoWordSubstring]]
) -> tuple[list[int], list[int], list[float]]:
    """Aligns and flattens model predictions with the markups.

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


def make_table(metrics: Metrics, floatf: str = '.2f', kind: TableFormat = 'simple') -> str:
    """Formats evaluation result as a table."""

    return tabulate(
        [[n, v] for n, v in metrics.items()],
        headers=['name', 'value'],
        tablefmt=kind,
        floatfmt=floatf,
    )


def equal_substrings(s1: SubstringLike, s2: SubstringLike) -> bool:
    """Compares two substring tuples by the first two elements (start, end)."""

    return s1[:2] == s2[:2]
