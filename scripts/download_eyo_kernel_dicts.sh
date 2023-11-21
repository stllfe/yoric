#!/bin/sh

set -e

mkdir -p data;
curl -Lo data/not-safe.txt https://raw.githubusercontent.com/e2yo/eyo-kernel/b7ee2c50268c0e0bfcfa5d36fe9da613485ee43d/dictionary/not_safe.txt
curl -Lo data/safe.txt https://raw.githubusercontent.com/e2yo/eyo-kernel/b7ee2c50268c0e0bfcfa5d36fe9da613485ee43d/dictionary/safe.txt
