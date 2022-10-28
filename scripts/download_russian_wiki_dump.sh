#!/bin/sh

set -e

mkdir -p data;
curl -Lo data/ruwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2;
