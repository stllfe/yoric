#!/bin/sh

set -e

mkdir -p data;
curl -Lo data/not-safe.txt https://raw.githubusercontent.com/e2yo/eyo-kernel/master/dict_src/not_safe.txt
curl -Lo data/safe.txt https://raw.githubusercontent.com/e2yo/eyo-kernel/master/dict_src/safe.txt
