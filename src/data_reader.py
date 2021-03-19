import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from .logger import logger
import sqlite3
import random
import csv
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = Path('data')
FOLD = 5
NELA_DIR = DATA_DIR / 'NELA'
Path(NELA_DIR).mkdir(parents=True, exist_ok=True)
NELA_FNAME = '{mode}_{fold}.tsv'
RESAMPLE_LIMIT = 100


def read_constraint_splits():
    train_fpath = DATA_DIR / 'train.tsv'
    val_fpath = DATA_DIR / 'val.tsv'
    test_fpath = DATA_DIR / 'test.tsv'
    train = pd.read_csv(train_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    val = pd.read_csv(val_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    test = pd.read_csv(test_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    return {
        'train': train,
        'val': val,
        'test': test
    }


def read_constraint(data_path):
    data_dir = Path(data_path)
    train_fpath = data_dir / 'train.tsv'
    val_fpath = data_dir / 'val.tsv'
    test_fpath = data_dir / 'groundtruth.tsv'
    train = pd.read_csv(train_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    val = pd.read_csv(val_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    test = pd.read_csv(test_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')

    return train[train.label == 'fake']['tweet'].tolist() + val[val.label == 'fake']['tweet'].tolist() + \
           test[test.label == 'fake']['tweet'].tolist()


def read_constraint(data_path, target='fake'):
    data_dir = Path(data_path)
    train_fpath = data_dir / 'train.tsv'
    val_fpath = data_dir / 'val.tsv'
    test_fpath = data_dir / 'groundtruth.tsv'
    train = pd.read_csv(train_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    val = pd.read_csv(val_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    test = pd.read_csv(test_fpath, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')

    print('Train set: ')
    print(train.groupby(['label']).count())
    print('Val set: ')
    print(val.groupby(['label']).count())
    print('Test set: ')
    print(test.groupby(['label']).count())

    return train[train.label == target]['tweet'].tolist() + val[val.label == target]['tweet'].tolist() + \
           test[test.label == target]['tweet'].tolist()


def read_coaid(data_path, target='fake'):
    data_dir = Path(data_path)
    train_fpath = data_dir / 'train.tsv'
    test_fpath = data_dir / 'test.tsv'
    val_fpath = data_dir / 'val.tsv'
    train = pd.read_csv(train_fpath, sep='\t')
    test = pd.read_csv(test_fpath, sep='\t')
    val = pd.read_csv(val_fpath, sep='\t')

    train.dropna(subset=['content'], inplace=True)
    test.dropna(subset=['content'], inplace=True)
    val.dropna(subset=['content'], inplace=True)

    print('Train set: ')
    print(train.groupby(['label']).count())
    print('Val set: ')
    print(val.groupby(['label']).count())
    print('Test set: ')
    print(test.groupby(['label']).count())


    return train[train['label'] == target]['content'].tolist() + test[test['label'] == target]['content'].tolist() + \
           val[val['label'] == target]['content'].tolist()


def read_covid_tweets(data_path, target='fake'):
    data_dir = Path(data_path)
    test_fpath = data_dir / 'test.tsv'
    test = pd.read_csv(test_fpath, sep='\t')

    print('Test set: ')
    print(test.groupby(['label']).count())


    return test[test['label'] == target]['content'].tolist()


def normalize(domain):
    domain = domain.strip()
    domain = domain.replace(' ', '')
    domain = domain.lower()
    return domain


def read_simplewiki(path: str):
    wiki = pd.read_csv(path, sep='\t')[['title', 'text']]
    title = wiki.title.map(lambda x: normalize(x)).to_list()
    text = wiki.text.map(lambda x: x.lower()).to_list()
    return dict(zip(title, text))


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


def handle_date(date):
    date = pd.to_datetime(date, errors='coerce')
    if type(date) == pd.NaT:
        return None
    else:
        return f"{date.year}-{date.month}-{date.day}"


def split_train_test(data, seed=None):
    print(len(data))
    if 'publish_date' in data.columns:
        data = data.sort_values(by='publish_date', ascending=True)
        test_index = round(len(data) * 0.6) + 1
        train_index = len(data) - test_index
        train = data[:train_index + 1]

        test = data.drop(train.index)
        val_index = test_index - test_index // 2
        # TODO add val split
        val = test[:val_index + 1]
        test = test[val_index + 1:]

    else:
        train, test = train_test_split(data, test_size=0.6, random_state=seed, stratify=data.label)
        val, test = train_test_split(test, test_size=0.5, random_state=seed, stratify=test.label)

    assert len(train) > len(test) and len(train) > len(val)
    assert len(train) + len(test) + len(val) == len(data)

    return train, val, test


def process_coaid(args):
    logger.info("Processing CoAID Dataset")
    coaid_dir = Path(args.path)
    doc_type = args.doc_type
    seed = args.seed

    coaid_all = []
    if doc_type == 'article':
        coaid_all = read_coaid_articles(coaid_dir)
        coaid_all = pd.concat(coaid_all)
        coaid_all = coaid_all[['url', 'content', 'title', 'label']]

    elif doc_type == 'claims':
        coaid_all = read_coaid_claims(coaid_dir)
        coaid_all = read_coaid_articles(coaid_dir)
        coaid_all = pd.concat(coaid_all)
        coaid_all['content'] = coaid_all['title']
        coaid_all = coaid_all[['url', 'content', 'title', 'label']]

    coaid_all.dropna(subset=['content'], inplace=True)
    coaid_all.drop_duplicates(inplace=True)

    train, val, test = split_train_test(coaid_all, seed)

    logger.info("Train Stats")
    logger.info(train.groupby(['label'])['title'].count())
    logger.info("Val Stats")
    logger.info(val.groupby(['label'])['title'].count())
    logger.info("Test Stats")
    logger.info(test.groupby(['label'])['title'].count())

    saved_path = coaid_dir / doc_type
    saved_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(saved_path / 'train.tsv', sep='\t', index=False)
    test.to_csv(saved_path / 'test.tsv', sep='\t', index=False)
    val.to_csv(saved_path / 'val.tsv', sep='\t', index=False)


def process_covid_tweet(args):
    logger.info("Processing Covid Tweet Dataset")
    covid_tweets = Path(args.path) / 'covid19_infodemic_english_data.tsv'

    tweets = pd.read_csv(covid_tweets, sep='\t')

    tweets_fake = tweets[(tweets['q2_label'] == '5_yes_definitely_contains_false_info') | (
            tweets['q2_label'] == '4_yes_probably_contains_false_info')]
    tweets_fake['label'] = 'fake'
    tweets_real = tweets[(tweets['q2_label'] == '2_no_probably_contains_no_false_info') | (
            tweets['q2_label'] == '1_no_definitely_contains_no_false_info')]
    tweets_real['label'] = 'true'

    test = pd.concat([tweets_real, tweets_fake])
    test.dropna(subset=['label'], inplace=True)
    test.drop_duplicates(inplace=True)
    test.rename(columns={"text": "content"},
                inplace=True)

    logger.info("Test Stats")
    logger.info(test.groupby(['label'])['content'].count())

    saved_path = Path(args.path)
    test.to_csv(saved_path / 'test.tsv', sep='\t', index=False)


def read_coaid_articles(coaid_dir):
    coaid_all = []
    for filepath in coaid_dir.rglob("NewsFakeCOVID-19.csv"):
        data = pd.read_csv(filepath, sep=',')
        data = data[data['type'] == 'article']
        data['publish_date'] = data.publish_date.apply(
            lambda x: handle_date(x))
        data["label"] = "fake"
        data.rename(columns={"news_url": "url"},
                    inplace=True)
        coaid_all.append(data)
    for filepath in coaid_dir.rglob("NewsRealCOVID-19.csv"):
        data = pd.read_csv(filepath, sep=',')
        data = data[data['type'] == 'article']
        data['publish_date'] = data.publish_date.apply(
            lambda x: handle_date(x))
        data["label"] = "true"
        data.rename(columns={"news_url": "url"},
                    inplace=True)
        coaid_all.append(data)
    return coaid_all


def read_coaid_claims(coaid_dir):
    coaid_all = []
    for filepath in coaid_dir.rglob("ClaimFakeCOVID-19.csv"):
        data = pd.read_csv(filepath, sep=',')
        data["label"] = "fake"
        data.rename(columns={"news_url": "url"},
                    inplace=True)
        coaid_all.append(data)
    for filepath in coaid_dir.rglob("ClaimRealCOVID-19.csv"):
        data = pd.read_csv(filepath, sep=',')
        data["label"] = "true"
        data.rename(columns={"news_url": "url"},
                    inplace=True)
        coaid_all.append(data)
    return coaid_all


def process_diabetes():
    pass


def process_recovery(args):
    recovery_path = Path(args.path)
    doc_type = args.doc_type
    if doc_type == 'article':
        recovery_dir = recovery_path / 'recovery-news-data.csv'
        data = pd.read_csv(recovery_dir, sep=',')
        data = data[['url', 'title', 'publisher', 'publish_date', 'body_text', 'reliability']]
        data.loc[data.reliability == 0, 'reliability'] = 'fake'
        data.loc[data.reliability == 1, 'reliability'] = 'true'
        data.rename(columns={"body_text": "content", "reliability": "label", "publisher": "source"}, inplace=True)
        data = data[['url', 'title', 'content', 'publish_date', 'label']]
        logger.info("Recovery News Stats")
        logger.info(data.groupby(['label'])['content'].count())
        data.dropna(subset=['content'], inplace=True)
        data.drop_duplicates(inplace=True)
        train, val, test = split_train_test(data)

    logger.info("Train Stats")
    logger.info(train.groupby(['label'])['title'].count())
    logger.info("Val Stats")
    logger.info(val.groupby(['label'])['title'].count())
    logger.info("Test Stats")
    logger.info(test.groupby(['label'])['title'].count())

    saved_path = recovery_path / doc_type
    saved_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(saved_path / 'train.tsv', sep='\t', index=False)
    val.to_csv(saved_path / 'val.tsv', sep='\t', index=False)
    test.to_csv(saved_path / 'test.tsv', sep='\t', index=False)


PROCESS_DATASETS = {
    'fakehealth': process_fakehealth,
    'healthreview': process_healthreview,
    'coaid': process_coaid,
    'recovery': process_recovery,
    'covid_tweets': process_covid_tweet
}

DATA_READER = {
    'constraint': read_constraint,
    'coaid': read_coaid,
    'recovery': read_coaid,
    'covid_tweets': read_covid_tweets
}

if __name__ == '__main__':
    parser = ArgumentParser()

    # NELA settings
    # parser.add_argument('--nela_2018', type=str)
    # parser.add_argument('--nela_2019', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--doc_type', type=str)
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    logger.info(f'Processing {args.data}')
    PROCESS_DATASETS[args.data](args)
