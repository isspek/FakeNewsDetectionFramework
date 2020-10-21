import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from .logger import logger
import sqlite3
import random

DATA_DIR = Path('data')
FOLD = 5
NELA_DIR = DATA_DIR / 'NELA'
Path(NELA_DIR).mkdir(parents=True, exist_ok=True)
NELA_FNAME = '{mode}_{fold}.tsv'


def read_constraint_splits():
    train_fpath = DATA_DIR / 'train.tsv'
    val_fpath = DATA_DIR / 'val.tsv'
    train = pd.read_csv(train_fpath, sep='\t')
    val = pd.read_csv(val_fpath, sep='\t')
    return {
        'train': train,
        'val': val
    }


def normalize_source(source: str):
    source = source.lower()
    source = source.strip()
    source = source.replace(" ", "")
    return source


def process_nela(nela_2018_path: str, nela_2019_path: str):
    '''
    NELA 2018 has the following metadata:
        - date
        - source
        - name (title)
        - content

    NELA 2019 has the following metadata:
        - id
        - date
        - source
        - title
        - content
        - author
        - url
        - published
        - published_utc
        - collection_utc

    We use aggregated labels from NELA 2019: unreliable: 2.0,  mixed: 1.0,  and  reliable: 0.0
    '''
    nela_2018_dir = Path(nela_2018_path)
    nela_2018_cnx = sqlite3.connect(nela_2018_dir / 'articles.db')
    nela_2018 = pd.read_sql_query("SELECT * FROM articles", nela_2018_cnx)
    nela_2018.rename(columns={"name": "title"}, inplace=True)
    # normalize sources of nela 2018
    nela_2018['source'] = nela_2018.source.apply(lambda x: normalize_source(x))

    nela_2019_dir = Path(nela_2019_path)
    nela_2019_cnx = sqlite3.connect(nela_2019_dir / 'nela-eng-2019.db')
    nela_2019 = pd.read_sql_query("SELECT * FROM newsdata", nela_2019_cnx)

    labels_path = nela_2019_dir / 'labels.csv'
    labels = pd.read_csv(labels_path)

    reliable_sources = labels[labels['aggregated_label'] == 0.0]['source'].unique()
    mixed_sources = labels[labels['aggregated_label'] == 1.0]['source'].unique()
    unreliable_sources = labels[labels['aggregated_label'] == 2.0]['source'].unique()
    satire_sources = labels[labels['Open Sources, satire'] >= 1.0]['source'].unique()

    reliable_nela_2019 = nela_2019[nela_2019.source.isin(reliable_sources)]
    mixed_nela_2019 = nela_2019[nela_2019.source.isin(mixed_sources)]
    unreliable_nela_2019 = nela_2019[nela_2019.source.isin(unreliable_sources)]
    satire_nela_2019 = nela_2019[nela_2019.source.isin(satire_sources)]

    reliable_nela_2018 = nela_2018[nela_2018.source.isin(reliable_sources)]
    mixed_nela_2018 = nela_2018[nela_2018.source.isin(mixed_sources)]
    unreliable_nela_2018 = nela_2018[nela_2018.source.isin(unreliable_sources)]
    satire_nela_2018 = nela_2018[nela_2018.source.isin(satire_sources)]

    reliable_merged = pd.concat([reliable_nela_2018, reliable_nela_2019])[['source', 'title', 'content']]
    reliable_merged['label'] = 'reliable'
    mixed_merged = pd.concat([mixed_nela_2018, mixed_nela_2019])[['source', 'title', 'content']]
    mixed_merged['label'] = 'mixed'
    unreliable_merged = pd.concat([unreliable_nela_2018, unreliable_nela_2019])[['source', 'title', 'content']]
    unreliable_merged['label'] = 'unreliable'
    satire_merged = pd.concat([satire_nela_2018, satire_nela_2019])[['source', 'title', 'content']]
    satire_merged['label'] = 'satire'
    reliable_merged = resample_sources(reliable_merged)
    mixed_merged = resample_sources(mixed_merged)
    unreliable_merged = resample_sources(unreliable_merged)
    satire_merged = resample_sources(satire_merged)

    # create 5-fold datasets
    for i in range(FOLD):
        test = []
        train = []

        _test, _train = split_by_source(reliable_merged, reliable_sources)
        train.append(_train)
        test.append(_test)

        _test, _train = split_by_source(mixed_merged, mixed_sources)
        train.append(_train)
        test.append(_test)

        _test, _train = split_by_source(unreliable_merged, unreliable_sources)
        train.append(_train)
        test.append(_test)

        _test, _train = split_by_source(satire_merged, satire_sources)
        train.append(_train)
        test.append(_test)

        train = pd.concat(train)
        test = pd.concat(test)
        train.to_csv(NELA_DIR / NELA_FNAME.format(mode='train', fold=i + 1), sep='\t', index=False)
        test.to_csv(NELA_DIR / NELA_FNAME.format(mode='test', fold=i + 1), sep='\t', index=False)

        logger.info(f'Info about the fold {i + 1}')
        logger.info(f'# of train samples {len(train)}')
        logger.info(f'# of test samples {len(test)}')


def resample_sources(sources):
    sources = sources.groupby('source').filter(lambda x: x['title'].count() > 9)
    sources.reset_index(drop=True, inplace=True)  # reset index
    sources = sources.groupby('source').apply(
        lambda x: x.sample(n=200) if len(x) > 200 else x).reset_index(drop=True)
    return sources


def split_by_source(data, sources):
    idx = round(0.8 * len(sources))
    if idx == len(sources) and idx != 0:
        idx = idx - 1

    random.shuffle(sources)
    train = data[data['source'].isin(sources[:idx])]
    test = data[data['source'].isin(sources[idx:])]
    assert train.source.unique() is not test.source.unique()
    return test, train


if __name__ == '__main__':
    parser = ArgumentParser()

    # NELA settings
    parser.add_argument('--nela_2018', type=str)
    parser.add_argument('--nela_2019', type=str)

    args = parser.parse_args()

    if args.nela_2018 and args.nela_2019:
        process_nela(args.nela_2018, args.nela_2019)
