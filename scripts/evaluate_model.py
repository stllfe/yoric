"""Script for evaluating model performance: quality metrics and inference speed."""

import argparse
import json

from dataclasses import asdict
from pathlib import Path
from typing import Union

from yoric import data
from yoric import models
from yoric import utils
from yoric.evaluate import evaluate_model
from yoric.evaluate import Evaluation
from yoric.model import YoModel


def build_model(config_path: str) -> YoModel:
    """Builds a :class:`YoModel` from the given config."""

    config = utils.load_yaml(config_path)
    config = config.get('model', config)
    params = config.get('params', {})

    cls: type[YoModel] = getattr(models, config['name'])

    if not issubclass(cls, YoModel):
        raise TypeError('Model is expected to be a subclass of YoModel!')  # noqa

    return cls(**params)


def save_evalution(evaluation: Evaluation, save_path: Union[str, Path]) -> None:
    with open(save_path, mode='w', encoding='utf-8') as fd:
        fd.write(json.dumps(asdict(evaluation), indent=2) + '\n')


def main(args: argparse.Namespace) -> None:
    print(f'Using config file: {args.config_path}')
    model = build_model(args.config_path)
    print(f'Model loaded: {type(model)}')

    dataset = data.load_dataset(args.markups_path, args.vocab_path)
    print(f'Using markups: {args.markups_path}')
    print(f'Using vocab: {args.vocab_path}')

    evaluation = evaluate_model(model, dataset, verbose=True)
    print(evaluation.table())

    if args.save_path:
        save_evalution(evaluation, args.save_path)
        print(f'Results saved: {args.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-path',
        type=Path,
        help='path to a config file with the model params',
        default='params.yaml',
    )
    parser.add_argument(
        '--markups-path',
        type=Path,
        help='path to a markups file',
        default='data/test-markups.jsonl.bz2',
    )
    parser.add_argument(
        '--vocab-path',
        type=Path,
        help='path to a vocab file',
        default='data/vocab.txt',
    )
    parser.add_argument(
        '--save-path', type=Path, help='path to save result in JSON format', required=False
    )
    main(parser.parse_args())
