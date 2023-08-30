#!/bin/sh

set -e

mkdir -p data;
curl -Lo data/ruwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/ruwiki/20230820/ruwiki-20230820-pages-articles-multistream.xml.bz2;
