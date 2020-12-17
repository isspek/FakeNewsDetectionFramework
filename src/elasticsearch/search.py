import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from src.data_reader import read_constraint_splits
from src.logger import logger
from .embedder import embed
from pathlib import Path

PREDICT_FILE_COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']
INDEX_NAME = 'vclaim'
LIMIT = 500


def create_connection(conn_string):
    logger.debug("Starting ElasticSearch client")
    try:
        es = Elasticsearch([conn_string], sniff_on_start=True)
    except:
        raise ConnectionError(f"Couldn't connect to Elastic Search instance at: {conn_string} \
                                Check if you've started it or if it listens on the port listed above.")
    logger.debug("Elasticsearch connected")
    return es


def clear_index(es):
    cleared = True
    try:
        es.indices.delete(index=INDEX_NAME)
    except:
        cleared = False
    return cleared


def build_index(es, data, index_file_path, fieldnames):
    vclaims_count = data.shape[0]
    clear_index(es)

    with open(index_file_path) as index_file:
        source = index_file.read().strip()
        es.indices.create(index=INDEX_NAME, body=source)
    logger.info(f"Building index of {vclaims_count} vclaims with fieldnames: {fieldnames}")
    for i, vclaim in tqdm(data.iterrows(), total=vclaims_count):
        if not es.exists(index=INDEX_NAME, id=i):
            body = vclaim.loc[fieldnames].to_dict()
            body['title_vector'] = embed(body['title'])
            body['content_vector'] = embed(body['content'])
            es.create(index=INDEX_NAME, id=i, body=body)


def build_query(tweet, search_keys, option):
    if option == 'semantic':
        query_vector = embed(tweet)
        query = {"query": {
            "script_score": {
                "query": {"multi_match": {"query": tweet, "fields": search_keys}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['content_vector']) + 1.0 + cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }}
    elif option == 'default':  # bm25
        query = {"query": {"multi_match": {"query": tweet, "fields": search_keys}}}
    else:
        raise Exception('Invalid query')
    return query


def get_score(es, tweet, search_keys, option, size=10):
    if len(tweet) > LIMIT: 
        tweet = tweet[:LIMIT]
    query = build_query(tweet, search_keys, option)
    try:
        response = es.search(index=INDEX_NAME, body=query, size=size)

    except:
        logger.error(f"No elasticsearch results for {tweet}")
        raise

    hits = response['hits']['hits']
    results = []
    for hit in hits:
        result = {}
        result['title'] = hit['_source']['title']
        result['content'] = hit['_source']['content']
        results.append(result)

    results = pd.DataFrame(results)
    return results


def get_results(es, tweets, option, search_keys, size):
    tweets_count = len(tweets)
    results = []
    for i, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        _results = get_score(es, tweet.tweet, option=option, search_keys=search_keys, size=size)
        _results['tweet_id'] = i
        results.append(_results)

    results = pd.concat(results)
    return results


def format_scores(scores):
    formatted_scores = []
    for tweet_id, s in scores.items():
        for vclaim_id, score in s.items():
            row = (str(tweet_id), 'Q0', str(vclaim_id), '1', str(score), 'elasic')
            formatted_scores.append(row)
    formatted_scores_df = pd.DataFrame(formatted_scores, columns=PREDICT_FILE_COLUMNS)
    return formatted_scores_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", "-k", default=['content', 'title'],
                        help="Keys to search in the document")
    parser.add_argument("--size", "-s", default=10,
                        help="Maximum results extracted for a query")
    parser.add_argument("--conn", "-c", default="127.0.0.1:9200",
                        help="HTTP/S URI to a instance of ElasticSearch")
    parser.add_argument("--option", type=str, choices=['default', 'semantic'])
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'])
    parser.add_argument("--index_file_path", type=str)
    return parser.parse_args()


def main(args):
    claims_dir = Path('data/processed')
    fakehealth = pd.read_csv(claims_dir / 'FakeHealth.tsv', sep='\t')
    fakehealth = fakehealth[fakehealth['label'] == 'fake']
    fakehealth = fakehealth.fillna('')  # replace Nan fields

    #TODO add more data
    # health_review = pd.read_csv(claims_dir / 'HealthReview.tsv', sep='\t')
    # previous_claims = pd.concat([fakehealth, health_review])

    previous_claims = fakehealth
    mode = args.mode

    data = read_constraint_splits()[mode]
    es = create_connection(args.conn)
    build_index(es, previous_claims, index_file_path=args.index_file_path, fieldnames=args.keys)

    output_dir = Path('data')
    results = get_results(es, data, option=args.option, search_keys=args.keys, size=args.size)
    results.to_csv(output_dir / f'{args.option}_{mode}_results.tsv', sep='\t', index=False)

    clear_index(es)


if __name__ == '__main__':
    args = parse_args()
    main(args)
