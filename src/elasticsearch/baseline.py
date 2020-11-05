import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.logger import logger
from src.data_reader import read_constraint_splits
from .embedder import embed

'''
The original of the script is: https://raw.githubusercontent.com/sshaar/clef2020-factchecking-task2/master/elastic_search_baseline.py
'''

PREDICT_FILE_COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']
INDEX_NAME = 'falsenews'
BATCH_SIZE = 1000


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
        es.indices.delete(index=INDEX_NAME, ignore=[404])
    except:
        cleared = False
    return cleared


def build_index(es, vclaims, fieldnames, index_file_path):
    vclaims_count = vclaims.shape[0]
    clear_index(es)

    # create an index based on index file
    with open(index_file_path) as index_file:
        source = index_file.read().strip()
        es.indices.create(index=INDEX_NAME, body=source)

    logger.info(f"Building index of {vclaims_count} false claims with fieldnames: {fieldnames}")
    docs = []
    for i, doc in tqdm(vclaims.iterrows(), total=vclaims_count):
        docs.append(doc)
        count = i + 1

        if count % BATCH_SIZE == 0:
            index_batch(docs)
            docs = []
            print("Indexed {} documents.".format(count))

    if docs:
        index_batch(docs, es)
        print("Indexed {} documents.".format(count))

    es.indices.refresh(index=INDEX_NAME)
    logger.info(f"{INDEX_NAME} has been successfully indexed.")


def index_batch(docs, es):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed(titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(es, requests)


def get_score(es, tweet, search_keys, size=10000):
    query = {"query": {"multi_match": {"query": tweet, "fields": search_keys}}}
    try:
        response = es.search(index=INDEX_NAME, body=query, size=size)
    except:
        logger.error(f"No elasticsearch results for {tweet}")
        raise
    results = response['hits']['hits']
    for result in results:
        info = result.pop('_source')
        result.update(info)
    df = pd.DataFrame(results)
    df['id'] = df._id.astype('int32').values
    df = df.set_index('id')
    return df._score


def get_scores(es, tweets, vclaims, search_keys, size):
    tweets_count, vclaims_count = len(tweets), len(vclaims)
    scores = {}

    logger.info(f"Geting RM5 scores for {tweets_count} tweets and {vclaims_count} vclaims")
    for i, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        score = get_score(es, tweet.tweet, search_keys=search_keys, size=size)
        scores[i] = score
    return scores


def format_scores(scores):
    formatted_scores = []
    for tweet_id, s in scores.items():
        for vclaim_id, score in s.items():
            row = (str(tweet_id), 'Q0', str(vclaim_id), '1', str(score), 'elastic')
            formatted_scores.append(row)
    formatted_scores_df = pd.DataFrame(formatted_scores, columns=PREDICT_FILE_COLUMNS)
    return formatted_scores_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vclaims", "-v", required=True,
                        help="TSV file with vclaims. Format: vclaim_id vclaim title")
    parser.add_argument("--tweets", "-t", required=True,
                        help="TSV file with tweets. Format: tweet_id tweet_content")
    parser.add_argument("--predict-file", "-p", required=True,
                        help="File in TREC Run format containing the model predictions")
    parser.add_argument("--keys", "-k", default=['content', 'title'], type=str,
                        help="Keys to search in the document", nargs='+')
    parser.add_argument("--size", "-s", default=10000,
                        help="Maximum results extracted for a query")
    parser.add_argument("--conn", "-c", default="127.0.0.1:9200",
                        help="HTTP/S URI to a instance of ElasticSearch")
    return parser.parse_args()


def main(args):
    fakehealth = pd.read_csv(args.vclaims, sep='\t')
    fakehealth = fakehealth[fakehealth['label'] == 'fake']
    fakehealth = fakehealth.fillna('')  # replace Nan fields
    data = read_constraint_splits()
    train = data['train']
    val = data['val']

    # todo: change this
    data = val

    es = create_connection(args.conn)
    build_index(es, fakehealth, fieldnames=args.keys)
    scores = get_scores(es, data, fakehealth, search_keys=args.keys, size=args.size)
    clear_index(es)

    formatted_scores = format_scores(scores)
    formatted_scores.to_csv(args.predict_file, sep='\t', index=False, header=False)
    logger.info(f"Saved scores from the model in file: {args.predict_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
