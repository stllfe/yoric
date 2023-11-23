"""Train script for word-level & sequence classification recurrent model."""

import gc
import os
import random

from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import cast, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import tyro

from dvclive import Live
from navec import Navec
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary
from tqdm import tqdm

from yoric import consts
from yoric import data
from yoric import evaluate
from yoric import utils
from yoric.models.wordrnn import config
from yoric.models.wordrnn import WordRNNModel
from yoric.models.wordrnn.data import batchify
from yoric.models.wordrnn.data import count_batches
from yoric.models.wordrnn.model import encode_word
from yoric.models.wordrnn.model import Samples
from yoric.models.wordrnn.model import StateDict
from yoric.models.wordrnn.model import WordBiLSTM
from yoric.models.wordrnn.tokenizer import Tokenizer


class Batch(NamedTuple):
    context: torch.Tensor
    lengths: torch.Tensor
    indices: torch.Tensor
    targets: torch.Tensor


def encode_batch_padded(
    samples: Samples,
    vocab: data.Vocab,
    padding_value: int = 0,
    device: Union[torch.device, str] = consts.CPU,
) -> Batch:
    """Converts raw markup samples to tensors of equal lengths."""

    context: list[torch.Tensor] = []
    lengths: list[int] = []
    indices: list[int] = []
    targets: list[int] = []

    for sample in samples:
        codes = torch.tensor([encode_word(word, vocab) for word in sample.words])
        lengths.append(len(codes))
        context.append(codes)
        indices.append(sample.index)
        targets.append(sample.target)

    return Batch(
        context=pad_sequence(context, batch_first=True, padding_value=padding_value).to(device),
        lengths=torch.tensor(lengths, device=device),
        indices=torch.tensor(indices, device=device),
        targets=torch.tensor(targets, device=device),
    )


def extract_pretrained_embeddings(navec: Navec) -> tuple[data.Vocab, npt.NDArray]:  # type: ignore
    """Extracts Navec vocab and embeddings to our classes."""

    # exclude <pad>, <unk> and english words
    navec_words = [w1 for w1 in navec.vocab.words[:-2] if not w1.isascii()]
    # we don't need words with Ё since our model would never see them in training
    # and won't be able to utilize them correctly at inference time anyway
    vocab_words = [utils.yeficate(w1) for w1 in navec_words]

    vocab = data.Vocab(vocab_words + [config.PAD, config.UNK])
    embs = np.zeros((len(vocab), *navec.pq.shape[1:]), dtype=np.float32)

    # move embeddings to correct positions
    # average word embeddings for Е and Ё variants
    w2v = {
        config.PAD: navec.get(config.PAD),
        config.UNK: navec.get(config.UNK),
    }
    ix = np.argsort(navec_words)
    for nv, vw in zip(np.take(navec_words, ix), np.take(vocab_words, ix)):
        if vw in w2v:
            w2v[vw] = (navec.get(nv) + w2v[vw]) / 2
        else:
            w2v[vw] = navec.get(nv)

    for i, w in enumerate(vocab):
        embs[i] = w2v[w]

    # todo: move to tests
    assert 'zzz' not in vocab
    assert 'всё' not in vocab
    assert np.array_equal(embs[vocab.get_label('все')], (navec.get('все') + navec.get('всё')) / 2)
    assert np.array_equal(embs[vocab.get_label(config.PAD)], navec.get(config.PAD))
    assert np.array_equal(embs[vocab.get_label(config.UNK)], navec.get(config.UNK))

    return vocab, embs


def make_progressbar(sampler: Iterable[Samples], limit: Optional[int] = None) -> Iterable[Samples]:
    """Wraps samples iterable with a progress bar."""

    return tqdm(
        islice(sampler, limit),
        desc='Train',
        unit='batch',
        leave=True,
        total=limit,
    )


def save_model_state(state: StateDict, path: Union[Path, str]) -> None:
    """Writes the model state to disk."""

    torch.save(state, path)


def fix_random_seed(seed: int) -> None:
    """Fixes seed for all random generators."""

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_current_lr(optimizer: optim.Optimizer) -> float:
    """Gets the current learning rate of optimizer."""

    for group in optimizer.param_groups:
        return cast(float, group['lr'])
    raise RuntimeError(f'No learning rate found for optimizer: {optimizer}')


def step(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, batch: Batch) -> float:
    """Runs a single training step.

    Returns:
        A detached float loss value.
    """

    optimizer.zero_grad()
    preds = model(batch.context, batch.lengths, batch.indices)

    loss = criterion(preds, batch.targets.float())
    loss.backward()

    optimizer.step()
    return cast(float, loss.detach().cpu().numpy().item())


def test(
    wordrnn: WordRNNModel, markups: Iterable[data.YeYoMarkup]
) -> tuple[list[int], list[int], list[float]]:
    """Runs model validation on the given data.

    Returns:
        A tuple of true, pred, score lists.
    """

    model = wordrnn.model
    training = model.training
    model.eval()

    X = [utils.yeficate(m.text) for m in markups]
    with torch.no_grad():
        outs = wordrnn.predict(X, verbose=True)
    t, p, s = evaluate.unroll_predictions(list(markups), outs)

    model.train(training)
    return t, p, s


def train(
    num_layers: int,
    dropout: float,
    num_epochs: int,
    batch_size: int,
    num_batches: Optional[int],
    learning_rate: float,
    weight_decay: float,
    navec_path: Union[Path, str],
    train_markups_path: Union[Path, str],
    test_markups_path: Union[Path, str],
    vocab_path: Union[Path, str],
    num_eval_samples: int = -1,
    schedule_patience: int = 10,
    schedule_factor: float = 0.1,
    stopping_patience: Optional[int] = None,
    device: Union[torch.device, str] = consts.CPU,
    exp_dir: Union[Path, str] = consts.MODEL_DIR,
    seed: int = 42,
) -> None:
    """Trains a word-level recurrent model for `Ё` restoration as a sequence classification."""

    # prepare experiment directory
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    fix_random_seed(seed)

    model_state_path = exp_dir / config.MODEL_FILE
    model_vocab_path = exp_dir / config.VOCAB_FILE

    vocab, embeddings = extract_pretrained_embeddings(Navec.load(navec_path))
    num_words, emb_dim = embeddings.shape

    print(f'Embeddings loaded: {num_words} x {emb_dim}')
    vocab.save(model_vocab_path)

    # init model and show the summary
    model = WordBiLSTM(
        num_words=num_words,
        emb_dim=emb_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_embeddings(torch.from_numpy(embeddings))
    model.to(device)

    print('Model initialized:')
    example = (
        torch.randint(low=0, high=num_words, size=(1, 50), dtype=torch.long),
        torch.tensor([50]),
        torch.tensor([1]),
    )
    summary(model, input_data=example, depth=1, device=device, verbose=1)

    # cleanup
    del embeddings
    del example
    gc.collect()

    # init the rest
    tokenizer = Tokenizer(lower=True)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=schedule_patience, factor=schedule_factor, verbose=True
    )

    # wrapper for existing inference and evaluation
    wordrnn = WordRNNModel(model=model, vocab=vocab, device=device)

    train_data = data.load_dataset(train_markups_path, vocab_path)
    test_data = data.load_dataset(test_markups_path, vocab_path)
    num_batches = num_batches or count_batches(train_data, batch_size=batch_size, drop_last=True)

    best_score = 0.0
    best_loss = np.inf
    plato_epochs = 0

    # run the train loop
    with Live(save_dvc_exp=True) as live:
        # run params
        live.log_param('seed', seed)
        live.log_param('num_epochs', num_epochs)
        live.log_param('num_batches', num_batches)
        live.log_param('num_eval_samples', num_eval_samples)
        live.log_param('learning_rate', learning_rate)
        live.log_param('weight_decay', weight_decay)
        live.log_param('schedule_patience', schedule_patience)
        live.log_param('schedule_factor', schedule_factor)

        # model hparams
        live.log_params(model.hparams)

        # model vocab
        live.log_artifact(model_vocab_path, type='vocab', desc='model vocab')

        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch:0>5}')
            model.train()

            sampler = batchify(
                train_data,
                tokenizer,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

            losses: list[float] = []
            for samples in make_progressbar(sampler, num_batches):
                batch = encode_batch_padded(
                    samples, vocab, padding_value=vocab.get_label(config.PAD), device=device
                )
                loss = step(model, criterion, optimizer, batch)
                losses.append(loss)

            epoch_loss = np.mean(losses).item()
            if epoch_loss <= best_loss:
                best_loss = epoch_loss

            t, p, s = test(wordrnn, test_data[:num_eval_samples])
            metrics = evaluate.Metrics.from_predictions(t, p, s)
            print(evaluate.make_table(metrics))

            for metric, value in metrics.items():
                live.log_metric(f'test/{metric}', value)

            live.log_metric('train/log_loss', epoch_loss)
            live.log_metric('train/lr', get_current_lr(optimizer))

            if metrics.f1_score >= best_score:
                best_score = metrics.f1_score
                plato_epochs = 0

                save_model_state(model.to_state(), model_state_path)
                live.log_artifact(model_state_path, type='model', desc='model checkpoint')
                print('Best checkpoint updated!')

                # log only the best model PR curve
                live.log_sklearn_plot('precision_recall', t, s, name='test/pr_curve')

            elif stopping_patience:
                plato_epochs += 1
                if plato_epochs > stopping_patience:
                    return print(
                        f'No improvement for {stopping_patience} epochs. Stopping early...'
                    )

            scheduler.step(metrics.f1_score)
            live.next_step()
            print(f'Loss: {epoch_loss:.4f} | F1: {metrics.f1_score:.4f} (best: {best_score:.4f})\n')


if __name__ == '__main__':
    tyro.cli(train)
