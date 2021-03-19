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
import json
import csv
from src.pk.topics_explorer import topic_embed
import gensim

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


def build_index_bio(es, previous_claims, bio_entities, index_file_path, fieldnames, topic_model, vocabulary):
    print('Semantic Bio Indexing')
    clear_index(es)
    assert len(previous_claims) == len(bio_entities)
    with open(index_file_path) as index_file:
        source = index_file.read().strip()
        es.indices.create(index=INDEX_NAME, body=source)

    for idx, (_, pclaim) in enumerate(tqdm(previous_claims.iterrows(), total=len(previous_claims))):
        if not es.exists(index=INDEX_NAME, id=idx):
            body = pclaim.loc[fieldnames].to_dict()
            body['topic_vector'] = topic_embed(text=body['content'], topic_model=topic_model,
                                               vocabulary=vocabulary).tolist()
            body['bio_ents'] = bio_entities[idx]
            body['content_vector'] = embed(body['content'])
            es.create(index=INDEX_NAME, id=idx, body=body)


def build_index_topic_semantic(es, previous_claims, index_file_path, fieldnames, topic_model, vocabulary):
    clear_index(es)
    with open(index_file_path) as index_file:
        source = index_file.read().strip()
        es.indices.create(index=INDEX_NAME, body=source)
    for idx, (_, pclaim) in enumerate(tqdm(previous_claims.iterrows(), total=len(previous_claims))):
        if not es.exists(index=INDEX_NAME, id=idx):
            body = pclaim.loc[fieldnames].to_dict()
            body['topic_vector'] = topic_embed(text=body['content'], topic_model=topic_model,
                                               vocabulary=vocabulary).tolist()
            body['content_vector'] = embed(body['content'])
            es.create(index=INDEX_NAME, id=idx, body=body)


def build_query(tweet, search_keys, option, entities=None, topic_model=None, vocabulary=None):
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

    elif option == 'semantic_bio':
        entities = entities[0]
        semantic_vector = embed(tweet)
        topic_vector = topic_embed(tweet, topic_model=topic_model, vocabulary=vocabulary).tolist()
        query = {"query": {
            "script_score": {
                "query": {
                    "terms": {
                        "bio_ents": entities,
                    }},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['content_vector']) +1.0 + cosineSimilarity(params.topic_vector, doc['topic_vector']) +1.0",
                    "params": {
                        "query_vector": semantic_vector,
                        "topic_vector": topic_vector
                    }
                }
            }
        }}

    elif option == 'topic_semantic':
        semantic_vector = embed(tweet)
        topic_vector = topic_embed(tweet, topic_model=topic_model, vocabulary=vocabulary).tolist()
        query = {"query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['content_vector']) +1.0 + cosineSimilarity(params.topic_vector, doc['topic_vector']) +1.0",
                    "params": {
                        "query_vector": semantic_vector,
                        "topic_vector": topic_vector
                    }
                }
            }
        }}
    else:
        raise Exception('Invalid query')
    return query


def get_score(es, tweet, search_keys, option, bio_entities, topic_model, vocabulary, size=10):
    try:
        if len(tweet) > LIMIT:
            tweet = tweet[:LIMIT]
        query = build_query(tweet, search_keys, option, entities=bio_entities, topic_model=topic_model,
                            vocabulary=vocabulary)
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
        if 'bio_ents' in hit['_source']:
            result['bio_ents'] = hit['_source']['bio_ents']
            result['bio_ents_query'] = bio_entities[0]
        result['query'] = tweet
        results.append(result)

    results = pd.DataFrame(results)
    return results


def get_results(es, samples, col_name, option, search_keys, size, bio_entities=None, topic_model=None, vocabulary=None):
    tweets_count = len(samples)
    results = []
    for i, sample in tqdm(samples.iterrows(), total=tweets_count):
        _results = get_score(es, sample[col_name], option=option, search_keys=search_keys, bio_entities=bio_entities,
                             topic_model=topic_model, vocabulary=vocabulary, size=size)
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
    parser.add_argument("--option", type=str, choices=['default', 'semantic', 'semantic_bio', 'topic_semantic'])
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'])
    parser.add_argument("--index_file_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--col_name", type=str)
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


def main(args):
    dir = Path('data')
    # TODO make it parametric
    fakehealth = pd.read_csv(dir / 'FakeHealth.tsv', sep='\t')
    fakehealth = fakehealth[fakehealth['label'] == 'fake']
    fakehealth = fakehealth.fillna('')  # replace Nan fields

    # TODO add more data
    # health_review = pd.read_csv(claims_dir / 'HealthReview.tsv', sep='\t')
    # previous_claims = pd.concat([fakehealth, health_review])

    previous_claims = fakehealth
    mode = args.mode
    data_path = Path(args.data_path)

    # data = read_constraint_splits()[mode]
    data_name = args.data
    output_dir = Path(args.output_dir)
    col_name = args.col_name

    if data_name == 'recovery':
        data = pd.read_csv(data_path, delimiter='\t')
    else:
        data = pd.read_csv(data_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                           delimiter='\t')

    es = create_connection(args.conn)
    logger.info(f'{args.option} has been selected.')

    if args.option == 'semantic_bio':
        topic_model_path = 'data/ffakenews_topics.model'
        topic_model = gensim.models.LdaModel.load(topic_model_path)
        vocabulary = topic_model.id2word
        src_bio_entities = json.load(open(dir / 'fakehealth_entities.json', 'r'))['bio_entities']
        build_index_bio(es, previous_claims, index_file_path=args.index_file_path, fieldnames=args.keys,
                        bio_entities=src_bio_entities,
                        topic_model=topic_model, vocabulary=vocabulary)
        target_bio_entities = json.load(open( Path(args.output_dir)  / f'{data_name}_{mode}_entities.json', 'r'))['bio_entities']
        results = get_results(es, data, col_name, option=args.option, search_keys=args.keys, size=args.size,
                              bio_entities=target_bio_entities,
                              topic_model=topic_model, vocabulary=vocabulary)
        results.to_csv(output_dir / f'{args.option}_{mode}_results.tsv', sep='\t', index=False)
    elif args.option == 'topic_semantic':
        topic_model_path = 'data/ffakenews_topics.model'
        topic_model = gensim.models.LdaModel.load(topic_model_path)
        vocabulary = topic_model.id2word
        build_index_topic_semantic(es, previous_claims, index_file_path=args.index_file_path, fieldnames=args.keys,
                                   topic_model=topic_model, vocabulary=vocabulary)
        results = get_results(es, data, col_name, option=args.option, search_keys=args.keys, size=args.size,
                              topic_model=topic_model, vocabulary=vocabulary)
        results.to_csv(output_dir / f'{args.option}_{mode}_results.tsv', sep='\t', index=False)
    else:
        build_index(es, previous_claims, index_file_path=args.index_file_path, fieldnames=args.keys)
        results = get_results(es, data, col_name, option=args.option, search_keys=args.keys, size=args.size)
        results.to_csv(output_dir / f'{data_name}_{args.option}_{mode}_results.tsv', sep='\t', index=False)

    clear_index(es)


if __name__ == '__main__':
    args = parse_args()
    main(args)
