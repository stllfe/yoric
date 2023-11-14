"""WordRNN related constants and configuration."""

from yoric import consts


UNK = '<unk>'
PAD = '<pad>'

MODEL_FILE = 'model.pt'
VOCAB_FILE = 'vocab.txt'

MODEL_PATH = consts.MODEL_DIR / MODEL_FILE
VOCAB_PATH = consts.MODEL_DIR / VOCAB_FILE
