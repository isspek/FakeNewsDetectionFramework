import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
from src.logger import logger
from cleantext import clean
import csv
import json
from tqdm import tqdm


def clean_helper(text):
    return clean(text,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 no_urls=True,  # replace all URLs with a special token
                 no_emails=True,
                 lower=True,
                 no_phone_numbers=True,
                 no_numbers=True,  # replace all numbers with a special token
                 no_digits=True,  # replace all digits with a special token
                 no_currency_symbols=True,
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="<DIGIT>",
                 replace_with_currency_symbol="<CUR>",
                 lang="en")


NUM_OF_PAST_CLAIMS = 10
NUM_OF_PAST_URLS = 5


class Constraint(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.labels2id = {'fake': 0, 'real': 1}
        self.id2labels = {val: key for key, val in self.labels2id.items()}

    def encode_for_transformer(self, tweets, claims=None):
        if claims:
            encoded_tweets = self.tokenizer(
                tweets,  # Sentence to encode.
                claims,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.hparams.max_seq_length,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt'  # Return pytorch tensors.
            )
        else:
            encoded_tweets = self.tokenizer(
                tweets,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.hparams.max_seq_length,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt'  # Return pytorch tensors.
            )

        input_ids = encoded_tweets['input_ids']
        attention_mask = encoded_tweets['attention_mask']
        return input_ids, attention_mask

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=RandomSampler(
                self.train_dataset),
            batch_size=self.hparams.train_batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.hparams.eval_batch_size,
            shuffle=False)


class Style(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        if self.hparams.train_path or self.hparams.do_train:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            logger.info(f'Total samples in training: {len(train_df)}')
            attention_mask, input_ids, labels = self.encode_style(train_df)
            self.train_dataset = TensorDataset(input_ids, attention_mask, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
            logger.info(f'Total samples in validation: {len(val_df)}')
            attention_mask, input_ids, labels = self.encode_style(val_df)
            self.val_dataset = TensorDataset(input_ids, attention_mask, labels)

    def encode_style(self, train_df):
        train_tweets = train_df.tweet.map(lambda x: clean_helper(x)).tolist()
        train_labels = train_df.label.tolist()
        input_ids, attention_mask, labels = self.encode_for_transformer(tweets=train_tweets, labels=train_labels)
        return attention_mask, input_ids, labels

    def encode_for_transformer(self, tweets, labels):
        encoded_tweets = self.tokenizer(
            tweets,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.hparams.max_seq_length,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )
        input_ids = encoded_tweets['input_ids']
        attention_mask = encoded_tweets['attention_mask']
        labels = torch.tensor([self.labels2id[i] for i in labels])

        return input_ids, attention_mask, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=RandomSampler(
                self.train_dataset),
            batch_size=self.hparams.train_batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.hparams.eval_batch_size,
            shuffle=False)


class History(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # tweets
        train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')

        # search results
        train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
        val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            tweet = row.tweet
            similar_false_claims = train_history_df[train_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                claims.append(torch.stack((input_ids, attention_mask)))
            if len(claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    claims.append(torch.stack((input_ids, attention_mask)))
            data.append(torch.stack(claims))
        data = torch.stack(data)
        train_labels = train_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # num of samples, num of evidences, [input ids, attention mask], additional dim, length
        self.train_dataset = TensorDataset(data, labels)

        data = []
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            tweet = row.tweet
            similar_false_claims = val_history_df[val_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                claims.append(torch.stack((input_ids, attention_mask)))
            if len(claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    claims.append(torch.stack((input_ids, attention_mask)))
            data.append(torch.stack(claims))
        data = torch.stack(data)
        val_labels = val_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in val_labels])

        self.val_dataset = TensorDataset(data, labels)


class Links(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reliability2id = {'reliable': [1, 0, 0, 0, 0], 'unreliable': [0, 1, 0, 0, 0], 'satire': [0, 0, 1, 0, 0],
                               'na': [0, 0, 0, 1, 0], '': [0, 0, 0, 0, 1]}
        self.suffix2id = {
            '': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'info': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'mp': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'co.uk': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'io': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ac.uk': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ca': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'com.au': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'media': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'gov.in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'org': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'edu': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'shop': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'edu.sg': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'int': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'govt.nz': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'tv': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'de': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'com': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'gov.ng': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'co': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ng': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'es': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'us': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'net': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'gov': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'report': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'org.uk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'biz': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'net.au': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    def setup(self, stage=None):
        # tweets
        train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')

        # link results
        train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
        val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            links = train_links[i]

            simple_wiki = []
            reliability = []
            suffix = []
            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack(suffix))
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        train_labels = train_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in train_labels])

        self.train_dataset = TensorDataset(simple_wiki_data, reliability_data, suffix_data, labels)

        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            links = val_links[i]

            simple_wiki = []
            reliability = []
            suffix = []
            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack(suffix))
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        val_labels = val_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in val_labels])

        self.val_dataset = TensorDataset(simple_wiki_data, reliability_data, suffix_data, labels)


class HistoryStyle(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # tweets
        train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')

        # search results
        train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
        val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        past_claims_data = []
        tweets_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            tweet = row.tweet
            similar_false_claims = train_history_df[train_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            tweets_data.append(torch.stack((input_ids, attention_mask)))
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))
            past_claims_data.append(torch.stack(past_claims))
        tweets_data = torch.stack(tweets_data)
        past_claims_data = torch.stack(past_claims_data)
        train_labels = train_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # num of samples, num of evidences, [input ids, attention mask], additional dim, length
        self.train_dataset = TensorDataset(past_claims_data, tweets_data, labels)

        past_claims_data = []
        tweets_data = []
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            tweet = row.tweet
            similar_false_claims = val_history_df[val_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            tweets_data.append(torch.stack((input_ids, attention_mask)))
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))
            past_claims_data.append(torch.stack(past_claims))
        tweets_data = torch.stack(tweets_data)
        past_claims_data = torch.stack(past_claims_data)
        val_labels = val_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in val_labels])

        self.val_dataset = TensorDataset(past_claims_data, tweets_data, labels)


class HistoryStyleLinks(Links):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # tweets
        train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')

        # search results
        train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
        val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        # link results
        train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
        val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']

        past_claims_data = []
        tweets_data = []
        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            links = train_links[i]
            simple_wiki = []
            reliability = []
            suffix = []
            tweet = row.tweet
            similar_false_claims = train_history_df[train_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            tweets_data.append(torch.stack((input_ids, attention_mask)))
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))
            past_claims_data.append(torch.stack(past_claims))

            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack((suffix)))

        tweets_data = torch.stack(tweets_data)
        past_claims_data = torch.stack(past_claims_data)
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        train_labels = train_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # num of samples, num of evidences, [input ids, attention mask], additional dim, length
        self.train_dataset = TensorDataset(past_claims_data, tweets_data, simple_wiki_data, reliability_data,
                                           suffix_data, labels)

        past_claims_data = []
        tweets_data = []
        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            links = val_links[i]
            simple_wiki = []
            reliability = []
            suffix = []
            tweet = row.tweet
            similar_false_claims = val_history_df[val_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            tweets_data.append(torch.stack((input_ids, attention_mask)))
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))
            past_claims_data.append(torch.stack(past_claims))

            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack((suffix)))

        tweets_data = torch.stack(tweets_data)
        past_claims_data = torch.stack(past_claims_data)
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        val_labels = val_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in val_labels])

        self.val_dataset = TensorDataset(past_claims_data, tweets_data, simple_wiki_data, reliability_data, suffix_data,
                                         labels)


class LinksStyle(Links):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # tweets
        train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')

        # search results
        train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
        val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        # link results
        train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
        val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']

        tweets_data = []
        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            links = train_links[i]
            simple_wiki = []
            reliability = []
            suffix = []
            tweet = row.tweet
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            tweets_data.append(torch.stack((input_ids, attention_mask)))

            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack((suffix)))

        tweets_data = torch.stack(tweets_data)
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        train_labels = train_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # num of samples, num of evidences, [input ids, attention mask], additional dim, length
        self.train_dataset = TensorDataset(tweets_data, simple_wiki_data, reliability_data,
                                           suffix_data, labels)

        tweets_data = []
        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            links = val_links[i]
            simple_wiki = []
            reliability = []
            suffix = []
            tweet = row.tweet
            similar_false_claims = val_history_df[val_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            tweets_data.append(torch.stack((input_ids, attention_mask)))
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))

            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack((suffix)))

        tweets_data = torch.stack(tweets_data)
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        val_labels = val_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in val_labels])

        self.val_dataset = TensorDataset(tweets_data, simple_wiki_data, reliability_data, suffix_data,
                                         labels)


class HistoryLinks(Links):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # tweets
        train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')

        # search results
        train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
        val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        # link results
        train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
        val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']

        past_claims_data = []
        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            links = train_links[i]
            simple_wiki = []
            reliability = []
            suffix = []
            tweet = row.tweet
            similar_false_claims = train_history_df[train_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))
            past_claims_data.append(torch.stack(past_claims))

            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack((suffix)))

        past_claims_data = torch.stack(past_claims_data)
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        train_labels = train_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # num of samples, num of evidences, [input ids, attention mask], additional dim, length
        self.train_dataset = TensorDataset(past_claims_data, simple_wiki_data, reliability_data,
                                           suffix_data, labels)

        past_claims_data = []
        simple_wiki_data = []
        reliability_data = []
        suffix_data = []
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            links = val_links[i]
            simple_wiki = []
            reliability = []
            suffix = []
            tweet = row.tweet
            similar_false_claims = val_history_df[val_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            cleaned_tweet = clean_helper(tweet)
            past_claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet, cleaned_claim)
                past_claims.append(torch.stack((input_ids, attention_mask)))

            if len(past_claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(past_claims)):
                    input_ids, attention_mask = self.encode_for_transformer('', cleaned_tweet)
                    past_claims.append(torch.stack((input_ids, attention_mask)))
            past_claims_data.append(torch.stack(past_claims))

            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer(link[''])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))

                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

                if 'suffix' in link:
                    suffix.append(torch.tensor(self.suffix2id[link['suffix']]))
                else:
                    suffix.append(torch.tensor(self.suffix2id['']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                    reliability.append(torch.tensor(self.reliability2id['']))
                    suffix.append(torch.tensor(self.suffix2id['']))
            simple_wiki_data.append(torch.stack(simple_wiki))
            reliability_data.append(torch.stack(reliability))
            suffix_data.append(torch.stack((suffix)))

        past_claims_data = torch.stack(past_claims_data)
        simple_wiki_data = torch.stack(simple_wiki_data)
        reliability_data = torch.stack(reliability_data)
        suffix_data = torch.stack(suffix_data)
        val_labels = val_df.label.tolist()
        labels = torch.tensor([self.labels2id[i] for i in val_labels])

        self.val_dataset = TensorDataset(past_claims_data, simple_wiki_data, reliability_data, suffix_data,
                                         labels)


DATA_MODELS = {
    'style': Style,
    'history': History,
    'links': Links,
    'history_style': HistoryStyle,
    'history_links': HistoryLinks,
    'history_links_style': HistoryStyleLinks,
    'links_style': LinksStyle
}
