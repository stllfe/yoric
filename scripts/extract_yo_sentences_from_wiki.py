"""A script that extracts the sentences with `Ё` letter from the Russian Wikipedia dump."""

import argparse
import logging

import razdel

from corus import load_wiki
from tqdm import tqdm


# suppress warnings from Wiki extractor
logging.getLogger().setLevel(logging.ERROR)


YO = 'ё'


def main(args: argparse.Namespace):
    wiki = load_wiki(args.wiki_path)

    count = 0
    with open(args.save_path, 'w', encoding='utf-8') as file, tqdm(
        wiki,
        desc='Extracting `Ё` sentences from wiki records',
        dynamic_ncols=True,
        colour='green',
    ) as progress:
        for record in progress:
            sentences = razdel.sentenize(record.text)
            for sentence in sentences:
                if YO in sentence.text.lower():
                    file.write(sentence.text + '\n')
                    count += 1
            progress.set_postfix(found=count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-w', '--wiki-path', 
        help='a path to wiki dump', 
        default='data/ruwiki-latest-pages-articles.xml.bz2'
    )
    parser.add_argument(
        '-f', '--save-path', 
        help='a filepath to save the sentences', 
        default='data/ruwiki-yo-sentences.txt'
    )
    main(parser.parse_args())
