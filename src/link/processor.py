from urlextract import URLExtract
from url_normalize import url_normalize
import requests
import tldextract
from src.data_reader import read_constraint_splits, read_nela_assessments, read_simplewiki
from src.logger import logger
from tqdm import tqdm
import re
import argparse
import json
from pathlib import Path
import pandas as pd

INVALID_URL = 'invalid_url'


class URLExtractor:
    def __init__(self):
        self.extractor = URLExtract()

    def normalize(self, url):
        if url.endswith('.'):
            url = url[: -1]
        try:
            session = requests.Session()
            resp = session.head(url, allow_redirects=True, timeout=5)
            unshorten_url = resp.url
            return url_normalize(unshorten_url)
        except requests.exceptions.RequestException as e:
            logger.error('Error {e} {url}'.format(e=e, url=url))
            return INVALID_URL
        return INVALID_URL

    def transform(self, post: str):
        normalized_url = []
        urls = self.extractor.find_urls(post)
        for url in urls:
            normalized_url.append(self.normalize(url))
        return normalized_url


def extract_url_metadata(url: str):
    res = tldextract.extract(url)
    return {'subdomain': res.subdomain,
            'domain': res.domain,
            'suffix': res.suffix,
            'alias': res.registered_domain}


def extract_urls(tweets, url_extractor):
    tweets_count = len(tweets)
    urls = {}
    for i, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        urls[i] = url_extractor.transform(tweet.tweet)
    return urls


USERNAME_REGEX = r"^https?:\/\/twitter.com\/(?:\#!\/)?(\w+)\/status\/\d+"


def extract_username_from_tweet_url(tweet_url):
    matches = re.finditer(USERNAME_REGEX, tweet_url, re.MULTILINE)

    for matchNum, match in enumerate(matches, start=1):
        return match.groups()[0]


def extract_domains(mode):
    output_dir = Path('data')
    links = json.load(open(output_dir / f'{mode}_links.json', 'r'))
    domains = []
    for tweet_id, link_arr in links.items():
        if len(link_arr) > 0:
            for link in link_arr:
                data = extract_url_metadata(link)
                domain = data['domain']

                if domain == 'twitter':
                    domain = extract_username_from_tweet_url(link).lower().strip()

                if domain not in domains:
                    domains.append(domain)

    domains = pd.DataFrame(domains, columns=['source'])
    domains.to_csv(output_dir / f'domains_{mode}.tsv', sep='\t', index=False)


def extract_features(mode, source_assessments, simple_wiki):
    '''
    This function encodes url engagements and save them to TODO
    '''
    output_dir = Path('data')
    links = json.load(open(output_dir / f'{mode}_links.json', 'r'))
    satire_sources = source_assessements[source_assessments['label'] == 'satire']['source'].unique()
    reliable_sources = source_assessements[source_assessments['label'] == 'reliable']['source'].unique()
    unreliable_sources = source_assessements[source_assessments['label'] == 'unreliable']['source'].unique()

    tweet_links = []
    for tweet_id, link_arr in links.items():
        processed_link = []
        if len(link_arr) > 0:
            for link in link_arr:
                data = extract_url_metadata(link)
                domain = data['domain']

                if domain == 'twitter':
                    if extract_username_from_tweet_url(link):
                        domain = extract_username_from_tweet_url(link).lower()
                        data['twitter_user'] = domain

                if domain == 'sky':
                    domain = 'skynews'

                logger.info(f'Requesting {domain}')

                data = add_reliability(data, domain, reliable_sources, satire_sources, unreliable_sources)
                data = add_simplewiki_description(data, domain, simple_wiki)
                processed_link.append(data)
            assert len(processed_link) == len(link_arr)
        tweet_links.append(processed_link)
    assert len(tweet_links) == len(links)
    tweet_links = {'links': tweet_links}
    output_dir = Path('data')
    json.dump(tweet_links, open(output_dir / f'{mode}_links_processed.json', 'w'))


def add_simplewiki_description(data, domain, simplewiki):
    if domain in simplewiki:
        data['simple_wiki'] = simplewiki[domain]
    else:
        data['simple_wiki'] = ''
    return data


def add_reliability(data, domain, reliable_sources, satire_sources, unreliable_sources):
    if domain in satire_sources:
        data['reliability'] = 'satire'
    elif domain in reliable_sources:
        data['reliability'] = 'reliable'
    elif domain in unreliable_sources:
        data['reliability'] = 'unreliable'
    else:
        data['reliability'] = 'na'
    return data


def extract_save_urls(mode):
    url_extractor = URLExtractor()
    data = read_constraint_splits()
    train_urls = extract_urls(data[mode], url_extractor)

    output_dir = Path('data')
    json.dump(train_urls, open(output_dir / f'{mode}_links.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", required=True,
                        help="Choose train, dev or test set", choices=['train', 'val', 'test'])
    parser.add_argument("--extract_link", action='store_true')
    parser.add_argument("--extract_features", action='store_true')
    parser.add_argument("--extract_domains", action='store_true')
    parser.add_argument("--nela_2019", type=str)
    parser.add_argument("--simple_wiki", type=str)
    args = parser.parse_args()

    if args.extract_link:
        extract_save_urls(args.mode)

    if args.extract_features:
        source_assessements = read_nela_assessments(args.nela_2019)
        simple_wiki = read_simplewiki(args.simple_wiki)
        extract_features(args.mode, source_assessments=source_assessements, simple_wiki=simple_wiki)

    if args.extract_domains:
        extract_domains(args.mode)
