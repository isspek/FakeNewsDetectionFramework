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

PREDICT_FILE_COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']
INDEX_NAME = 'vclaim'


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
    logger.info(f"Builing index of {vclaims_count} vclaims with fieldnames: {fieldnames}")
    for i, vclaim in tqdm(data.iterrows(), total=vclaims_count):
        if not es.exists(index=INDEX_NAME, id=i):
            body = vclaim.loc[fieldnames].to_dict()
            body['title_vector'] = embed(body['title'])
            body['content_vector'] = embed(body['content'])
            es.create(index=INDEX_NAME, id=i, body=body)


def get_score(es, tweet, search_keys, size=10):
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
        print(info)
    print(len(results))
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
            row = (str(tweet_id), 'Q0', str(vclaim_id), '1', str(score), 'elasic')
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
    parser.add_argument("--keys", "-k", default=['content', 'title'],
                        help="Keys to search in the document")
    parser.add_argument("--size", "-s", default=10,
                        help="Maximum results extracted for a query")
    parser.add_argument("--conn", "-c", default="127.0.0.1:9200",
                        help="HTTP/S URI to a instance of ElasticSearch")
    parser.add_argument("--index_file_path", type=str)
    return parser.parse_args()


def main(args):
    fakehealth = pd.read_csv(args.vclaims, sep='\t')
    fakehealth = fakehealth[fakehealth['label'] == 'fake']
    fakehealth = fakehealth.fillna('')  # replace Nan fields
    data = read_constraint_splits()
    train = data['train']
    val = data['val']

    # todo: change this
    data = val[:1]

    es = create_connection(args.conn)
    build_index(es, fakehealth, index_file_path=args.index_file_path, fieldnames=args.keys)
    scores = get_scores(es, data, fakehealth, search_keys=args.keys, size=args.size)
    # clear_index(es)

    # formatted_scores = format_scores(scores)
    # formatted_scores.to_csv(args.predict_file, sep='\t', index=False, header=False)
    # logger.info(f"Saved scores from the model in file: {args.predict_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
