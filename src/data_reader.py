import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from .logger import logger
import sqlite3
import random
import csv
import json
from tqdm import tqdm

DATA_DIR = Path('data')
FOLD = 5
NELA_DIR = DATA_DIR / 'NELA'
Path(NELA_DIR).mkdir(parents=True, exist_ok=True)
NELA_FNAME = '{mode}_{fold}.tsv'
RESAMPLE_LIMIT = 100


def read_constraint_splits():
    train_fpath = DATA_DIR / 'train.tsv'
    val_fpath = DATA_DIR / 'val.tsv'
    train = pd.read_csv(train_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    val = pd.read_csv(val_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    return {
        'train': train,
        'val': val
    }


def normalize_source(source: str):
    source = source.lower()
    source = source.strip()
    source = source.replace(" ", "")
    return source


def read_nela_assessments(nela_2019_path):
    nela_2019_dir = Path(nela_2019_path)
    labels_path = nela_2019_dir / 'labels.csv'
    labels = pd.read_csv(labels_path)

    reliable_sources = labels[labels['aggregated_label'] == 0.0]['source'].unique()
    reliable_sources = pd.DataFrame(reliable_sources, columns=['source'])
    reliable_sources['label'] = 'reliable'
    unreliable_sources = labels[labels['aggregated_label'] == 2.0]['source'].unique()
    unreliable_sources = pd.DataFrame(unreliable_sources, columns=['source'])
    unreliable_sources['label'] = 'unreliable'
    satire_sources = labels[labels['Media Bias / Fact Check, label'] == 'satire']['source'].unique()
    satire_sources = pd.DataFrame(satire_sources, columns=['source'])
    satire_sources['label'] = 'satire'

    source_assessments = pd.concat([reliable_sources, unreliable_sources, satire_sources])
    return source_assessments


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
    # nela_2018_dir = Path(nela_2018_path)
    # nela_2018_cnx = sqlite3.connect(nela_2018_dir / 'articles.db')
    # nela_2018 = pd.read_sql_query("SELECT * FROM articles", nela_2018_cnx)
    # nela_2018.rename(columns={"name": "title"}, inplace=True)
    # # normalize sources of nela 2018
    # nela_2018['source'] = nela_2018.source.apply(lambda x: normalize_source(x))

    nela_2019_dir = Path(nela_2019_path)
    nela_2019_cnx = sqlite3.connect(nela_2019_dir / 'nela-eng-2019.db')
    nela_2019 = pd.read_sql_query("SELECT * FROM newsdata", nela_2019_cnx)

    labels_path = nela_2019_dir / 'labels.csv'
    labels = pd.read_csv(labels_path)

    reliable_sources = labels[labels['aggregated_label'] == 0.0]['source'].unique()
    # mixed_sources = labels[labels['aggregated_label'] == 1.0]['source'].unique()
    unreliable_sources = labels[labels['aggregated_label'] == 2.0]['source'].unique()
    satire_sources = labels[labels['Media Bias / Fact Check, label'] == 'satire']['source'].unique()

    reliable_nela_2019 = nela_2019[nela_2019.source.isin(reliable_sources)]
    # mixed_nela_2019 = nela_2019[nela_2019.source.isin(mixed_sources)]
    unreliable_nela_2019 = nela_2019[nela_2019.source.isin(unreliable_sources)]
    satire_nela_2019 = nela_2019[nela_2019.source.isin(satire_sources)]

    # reliable_nela_2018 = nela_2018[nela_2018.source.isin(reliable_sources)]
    # mixed_nela_2018 = nela_2018[nela_2018.source.isin(mixed_sources)]
    # unreliable_nela_2018 = nela_2018[nela_2018.source.isin(unreliable_sources)]
    # satire_nela_2018 = nela_2018[nela_2018.source.isin(satire_sources)]

    # reliable_merged = pd.concat([reliable_nela_2018, reliable_nela_2019])[['source', 'title', 'content']]
    # reliable_merged['label'] = 'reliable'
    # mixed_merged = pd.concat([mixed_nela_2018, mixed_nela_2019])[['source', 'title', 'content']]
    # mixed_merged['label'] = 'mixed'
    # unreliable_merged = pd.concat([unreliable_nela_2018, unreliable_nela_2019])[['source', 'title', 'content']]
    # unreliable_merged['label'] = 'unreliable'
    # satire_merged = pd.concat([satire_nela_2018, satire_nela_2019])[['source', 'title', 'content']]
    reliable_merged = reliable_nela_2019[['source', 'title', 'content']]
    reliable_merged['label'] = 'reliable'
    # mixed_merged = mixed_nela_2019[['source', 'title', 'content']]
    # mixed_merged['label'] = 'mixed'
    unreliable_merged = unreliable_nela_2019[['source', 'title', 'content']]
    unreliable_merged['label'] = 'unreliable'
    satire_merged = satire_nela_2019[['source', 'title', 'content']]
    satire_merged['label'] = 'satire'
    reliable_merged = resample_sources(reliable_merged)
    # mixed_merged = resample_sources(mixed_merged)
    unreliable_merged = resample_sources(unreliable_merged)
    satire_merged = resample_sources(satire_merged)

    # create 5-fold datasets
    for i in range(FOLD):
        test = []
        train = []

        _test, _train = split_by_source(reliable_merged, reliable_sources)
        train.append(_train)
        test.append(_test)

        # _test, _train = split_by_source(mixed_merged, mixed_sources)
        # train.append(_train)
        # test.append(_test)

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

        logger.info(train.groupby(['label'])['title'].count())
        logger.info(test.groupby(['label'])['title'].count())


def resample_sources(sources):
    sources = sources.groupby('source').filter(lambda x: x['title'].count() > 9)
    sources.reset_index(drop=True, inplace=True)  # reset elasticsearch
    sources = sources.groupby('source').apply(
        lambda x: x.sample(n=RESAMPLE_LIMIT) if len(x) > RESAMPLE_LIMIT else x).reset_index(drop=True)
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


def process_fakehealth(dataset_path: str):
    def extract_content(dir, news_source, news_ids, label):
        processed_data = []
        for news_id in news_ids:
            data = {}
            fname = f"content/{news_source}/{news_id}.json"
            if not (dir / fname).exists():
                logger.error(f"{news_id} does not exist")
                continue
            with open(dir / fname) as f:
                news_item = json.load(f)
                data["content"] = news_item["text"]
                data["url"] = news_item["url"]
                data["title"] = news_item["title"]
                data["news_id"] = news_id
                data["label"] = label
                processed_data.append(data)
        logger.info(f"Num of processed {label} news with date {len(processed_data)}")
        return processed_data

    dir = Path(dataset_path)
    reviews = pd.read_json(dir / 'reviews/HealthStory.json')
    score = 3
    fake_stories = reviews[reviews["rating"] < score]
    true_stories = reviews[reviews["rating"] >= score]

    logger.info("Stats of Health Story")
    logger.info(f"Num of fake stories {len(fake_stories)}")
    logger.info(f"Num of true stories {len(true_stories)}")
    logger.info(f"Num of unique sources in fake stories {len(fake_stories.news_source.unique())}")
    logger.info(f"Num of unique sources in true stories {len(true_stories.news_source.unique())}")
    logger.info(
        f"Num of overlapped sources {len(set(fake_stories.news_source.unique()) - set(true_stories.news_source.unique()))}")

    filtered_fake_stories = pd.DataFrame(extract_content(dir, "HealthStory", fake_stories.news_id.values, "fake"))
    filtered_true_stories = pd.DataFrame(extract_content(dir, "HealthStory", true_stories.news_id.values, "true"))

    releases = pd.read_json(dir / 'reviews/HealthRelease.json')
    fake_releases = releases[releases["rating"] < score]
    true_releases = releases[releases["rating"] >= score]

    logger.info("Stats of Health Releases")
    logger.info(f"Num of fake releases {len(fake_stories)}")
    logger.info(f"Num of true releases {len(true_stories)}")
    logger.info(f"Num of unique sources in fake releases {len(fake_releases.news_source.unique())}")
    logger.info(f"Num of unique sources in true releases {len(true_releases.news_source.unique())}")
    logger.info(
        f"Num of overlapped sources {len(set(fake_stories.news_source.unique()) - set(true_stories.news_source.unique()))}")

    filtered_fake_releases = pd.DataFrame(extract_content(dir, "HealthRelease", fake_releases.news_id.values, "fake"))
    filtered_true_releases = pd.DataFrame(extract_content(dir, "HealthRelease", true_releases.news_id.values, "true"))
    merged_data = pd.concat(
        [filtered_fake_releases, filtered_true_stories, filtered_fake_stories, filtered_true_releases])

    output_dir = 'data/processed'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    processed_dir = Path(output_dir)
    processed_file = processed_dir / 'FakeHealth.tsv'
    merged_data.to_csv(processed_file, sep='\t', index=False)
    logger.info(f"Merged data is saved to {processed_file}")


def process_healthreview(dataset_path: str):
    data = pd.read_csv(dataset_path)
    data = data[
        (data['temp_label'] == 'false') | (
                data['temp_label'] == 'mostly_false')]  # select only false claims for indexing
    latest_date = max(data['claimReview_datePublished'].tolist())
    earliest_date = min(data['claimReview_datePublished'].tolist())
    logger.info(f'Latest review id {latest_date}')
    logger.info(f'Earliest review id {earliest_date}')
    processed_data = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if not 'health' in row.claimReview_url or not 'health' in row.claimReview_claimReviewed:
            continue
        sample = {}
        sample['content'] = row.loc['claimReview_claimReviewed']
        sample['title'] = row.loc['extra_title']
        sample['label'] = row.loc['temp_label']
        sample['news_id'] = ''
        sample['url'] = ''
        processed_data.append(sample)

    processed_data = pd.DataFrame(processed_data)
    logger.info(f'Num of samples in processed data {len(processed_data)}')
    output_dir = 'data/processed'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    processed_dir = Path(output_dir)
    processed_file = processed_dir / 'HealthReview.tsv'
    processed_data.to_csv(processed_file, sep='\t', index=False)
    logger.info(f"Merged data is saved to {processed_file}")


if __name__ == '__main__':
    parser = ArgumentParser()

    # NELA settings
    parser.add_argument('--nela_2018', type=str)
    parser.add_argument('--nela_2019', type=str)
    parser.add_argument('--fakehealth', type=str)
    parser.add_argument('--healthreview', type=str)

    args = parser.parse_args()

    if args.nela_2018 and args.nela_2019:
        process_nela(args.nela_2018, args.nela_2019)

    if args.fakehealth:
        logger.info('Processing Fakehealth datasets...')
        process_fakehealth(args.fakehealth)

    if args.healthreview:
        logger.info('Processing Healthreview dataset...')
        process_healthreview(args.healthreview)
