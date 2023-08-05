"""Model performance evaluation stuff."""

import re
import time

from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass

from sklearn import metrics
from tqdm import tqdm

from yoric import utils
from yoric.data import YeYoDataset
from yoric.model import YoModel
from yoric.model import YoWordSubstrings


@dataclass
class EvaluateResult:
    """Model evaluation result."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fh_score: float
    auroc: float
    wall_time: float
    cpu_time: float
    tokens_per_sec: float

    def __format__(self, fs: str) -> str:
        return '\n'.join([f'{m}: {format(v, fs)}' for m, v in asdict(self).items()])

    def __str__(self) -> str:
        return self.__format__('')


def get_tokens_count(text: str) -> int:
    """Calculates count of tokens in a text."""

    return len(list(re.finditer(utils.WORDS_REGEX, text)))


def evaluate_model(model: YoModel, dataset: YeYoDataset, verbose: bool = False) -> EvaluateResult:
    """Runs metrics calculation on the given model and dataset."""

    X = [utils.yeficate(markup.text) for markup in dataset]

    wall_time = time.perf_counter()
    cpu_time = time.process_time()

    y_pred: Iterable[YoWordSubstrings] = model.predict(X, verbose=verbose)

    wall_time = time.perf_counter() - wall_time
    cpu_time = time.process_time() - cpu_time

    if verbose:
        y_pred = tqdm(y_pred, desc='Prepare labels')

    y_true_binary: list[int] = []
    y_pred_binary: list[int] = []

    for markup, y_hat in zip(dataset, y_pred):
        y_true_binary.extend(markup.targets)
        for span in markup.spans:
            y_pred_binary.append(1 if span in y_hat else 0)

    num_tokens = sum(map(get_tokens_count, X))

    result = EvaluateResult(
        accuracy=float(metrics.accuracy_score(y_true_binary, y_pred_binary)),
        precision=float(metrics.precision_score(y_true_binary, y_pred_binary)),
        recall=float(metrics.recall_score(y_true_binary, y_pred_binary)),
        f1_score=float(metrics.f1_score(y_true_binary, y_pred_binary)),
        fh_score=float(metrics.fbeta_score(y_true_binary, y_pred_binary, beta=0.5)),
        auroc=float(metrics.roc_auc_score(y_true_binary, y_pred_binary)),
        wall_time=wall_time,
        cpu_time=cpu_time,
        tokens_per_sec=num_tokens / wall_time,
    )

    if verbose:
        print(f'\n{result:.4f}\n')

    return result
