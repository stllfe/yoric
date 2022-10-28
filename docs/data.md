# Данные для обучения

Модель обучена на данных из русской Википедии. Скрипты для получения данных находятся в директории `scripts`. Текущий датасет собран следующей командой:
```shell
python scripts/extract_sentences_from_wiki.py -j 8 -s 50 -n 1000000
```
Затем произведена первичная очистка текста:
```shell
python scripts/preprocess_sentences.py -j 4
```
Финальный датасет собирается следующей командой:
```shell
python scripts/compile_dataset.py
```
