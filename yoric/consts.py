"""Project-wide constants and configuration."""

from pathlib import Path


PROJ_DIR = Path(__file__).parent.parent
DATA_DIR = PROJ_DIR / 'data'
MODEL_DIR = PROJ_DIR / 'model'

NOT_SAFE_DICT_PATH = DATA_DIR / 'not-safe.txt'
SAFE_DICT_PATH = DATA_DIR / 'safe.txt'
