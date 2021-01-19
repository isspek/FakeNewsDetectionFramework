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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=self.hparams.eval_batch_size,
            shuffle=False)

    def encode_history(self, train_df, train_history_df):
        past = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            similar_false_claims = train_history_df[train_history_df['tweet_id'] == i]
            similar_false_claims = similar_false_claims.fillna('')
            similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()
            claims = []
            for claim in similar_false_claims:
                cleaned_claim = clean_helper(claim)
                input_ids, attention_mask = self.encode_for_transformer(cleaned_claim)
                claims.append(torch.stack((input_ids, attention_mask)))
            if len(claims) < NUM_OF_PAST_CLAIMS:
                for _ in range(NUM_OF_PAST_CLAIMS - len(claims)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    claims.append(torch.stack((input_ids, attention_mask)))
            past.append(torch.stack(claims))
        past = torch.stack(past)
        return past

    def encode_post(self, train_df):
        post = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            tweet = row.tweet
            cleaned_tweet = clean_helper(tweet)
            input_ids, attention_mask = self.encode_for_transformer(cleaned_tweet)
            post.append(torch.stack((input_ids, attention_mask)))
        post = torch.stack(post)
        return post

    def extract_simple_wiki_feature(self, train_df, train_links):
        simple_wiki_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            links = train_links[i]
            simple_wiki = []
            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'simple_wiki' in link:
                    input_ids, attention_mask = self.encode_for_transformer(link['simple_wiki'])
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
                else:
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    input_ids, attention_mask = self.encode_for_transformer('')
                    simple_wiki.append(torch.stack((input_ids, attention_mask)))
            simple_wiki_data.append(torch.stack(simple_wiki))
        simple_wiki_data = torch.stack(simple_wiki_data)
        return simple_wiki_data

    def extract_reliability_feature(self, train_df, train_links):
        reliability_data = []
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            links = train_links[i]
            reliability = []
            for idx, link in enumerate(links):
                if idx == NUM_OF_PAST_URLS:
                    break
                if 'reliability' in link:
                    reliability.append(torch.tensor(self.reliability2id[link['reliability']]))
                else:
                    reliability.append(torch.tensor(self.reliability2id['na']))

            if len(links) < NUM_OF_PAST_URLS:
                for _ in range(NUM_OF_PAST_URLS - len(links)):
                    reliability.append(torch.tensor([0, 0, 0, 0]))
            reliability_data.append(torch.stack(reliability))
        reliability_data = torch.stack(reliability_data)
        return reliability_data



class Style(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if self.hparams.train_path:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            post_data = self.encode_post(train_df)
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])
            self.train_dataset = TensorDataset(post_data, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                 delimiter='\t')
            post_data = self.encode_post(val_df)
            val_labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in val_labels])
            self.val_dataset = TensorDataset(post_data, labels)

        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                  delimiter='\t')
            post_data = self.encode_post(test_df)[2130:2132]
            self.test_dataset = TensorDataset(post_data)

class History(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if self.hparams.train_path or self.hparams.do_train:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
            logger.info(f'Total samples in training: {len(train_df)}')
            past = self.encode_history(train_df, train_history_df)
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])
            post = self.encode_post(train_df)
            self.train_dataset = TensorDataset(past, post, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                 delimiter='\t')
            val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')
            past = self.encode_history(val_df, val_history_df)
            labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in labels])
            post = self.encode_post(val_df)
            self.val_dataset = TensorDataset(past, post, labels)

        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                  delimiter='\t')
            test_history_df = pd.read_csv(self.hparams.history_test_path, sep='\t')
            past = self.encode_history(test_df, test_history_df)
            post = self.encode_post(test_df)
            self.test_dataset = TensorDataset(past, post)

class Links(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reliability2id = {'reliable': [1, 0, 0, 0], 'unreliable': [0, 1, 0, 0], 'satire': [0, 0, 1, 0],
                               'na': [0, 0, 0, 1]}
        self.wiki = self.hparams.wiki
        self.reliability = self.hparams.reliability

    def setup(self, stage=None):
        if self.hparams.train_path or self.hparams.do_train:
            # tweets
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            # link results
            train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])

            if self.wiki:
                wiki_data = self.extract_simple_wiki_feature(train_df, train_links)
                self.train_dataset = TensorDataset(wiki_data, labels)
            elif self.reliability:
                reliability_data = self.extract_reliability_feature(train_df, train_links)
                self.train_dataset = TensorDataset(reliability_data, labels)
            else:
                wiki_data = self.extract_simple_wiki_feature(train_df, train_links)
                reliability_data = self.extract_reliability_feature(train_df, train_links)
                self.train_dataset = TensorDataset(wiki_data, reliability_data, labels)
        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
            val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']
            val_labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in val_labels])
            if self.wiki:
                wiki_data = self.extract_simple_wiki_feature(val_df, val_links)
                self.val_dataset = TensorDataset(wiki_data, labels)
            elif self.reliability:
                reliability_data = self.extract_reliability_feature(val_df, val_links)
                self.val_dataset = TensorDataset(reliability_data, labels)
            else:
                wiki_data = self.extract_simple_wiki_feature(val_df, val_links)
                reliability_data = self.extract_reliability_feature(val_df, val_links)
                self.val_dataset = TensorDataset(wiki_data, reliability_data, labels)
        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False, delimiter='\t')
            test_links = json.load(open(self.hparams.link_test_path, 'r'))['links']
            if self.wiki:
                wiki_data = self.extract_simple_wiki_feature(test_df, test_links)
                self.test_dataset = TensorDataset(wiki_data)
            elif self.reliability:
                reliability_data = self.extract_reliability_feature(test_df, test_links)
                self.test_dataset = TensorDataset(reliability_data)
            else:
                wiki_data = self.extract_simple_wiki_feature(test_df, test_links)
                reliability_data = self.extract_reliability_feature(test_df, test_links)
                self.test_dataset = TensorDataset(wiki_data, reliability_data)


class HistoryStyle(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if self.hparams.train_path:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
            post_data = self.encode_post(train_df)
            history = self.encode_history(train_df, train_history_df)
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])
            self.train_dataset = TensorDataset(history, post_data, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                 delimiter='\t')
            val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')
            post_data = self.encode_post(val_df)
            history = self.encode_history(val_df, val_history_df)
            val_labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in val_labels])
            self.val_dataset = TensorDataset(history, post_data, labels)

        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                  delimiter='\t')
            test_history_df = pd.read_csv(self.hparams.history_test_path, sep='\t')
            post_data = self.encode_post(test_df)
            history = self.encode_history(test_df, test_history_df)
            self.test_dataset = TensorDataset(history, post_data)


class HistoryStyleLinks(Links):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if self.hparams.train_path:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
            train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
            simple_wiki_data = self.extract_simple_wiki_feature(train_df, train_links)
            reliability_data = self.extract_reliability_feature(train_df, train_links)
            post_data = self.encode_post(train_df)
            history = self.encode_history(train_df, train_history_df)
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])
            self.train_dataset = TensorDataset(history, post_data, simple_wiki_data, reliability_data, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                 delimiter='\t')
            val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']
            val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')
            simple_wiki_data = self.extract_simple_wiki_feature(val_df, val_links)
            reliability_data = self.extract_reliability_feature(val_df, val_links)
            post_data = self.encode_post(val_df)
            history = self.encode_history(val_df, val_history_df)
            val_labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in val_labels])
            self.val_dataset = TensorDataset(history, post_data, simple_wiki_data, reliability_data, labels)

        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                  delimiter='\t')
            test_links = json.load(open(self.hparams.link_test_path, 'r'))['links']
            test_history_df = pd.read_csv(self.hparams.history_test_path, sep='\t')
            simple_wiki_data = self.extract_simple_wiki_feature(test_df, test_links)
            reliability_data = self.extract_reliability_feature(test_df, test_links)
            post_data = self.encode_post(test_df)
            history = self.encode_history(test_df, test_history_df)
            self.test_dataset = TensorDataset(history, post_data, simple_wiki_data, reliability_data)


class LinksStyle(Links):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if self.hparams.train_path:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
            simple_wiki_data = self.extract_simple_wiki_feature(train_df, train_links)
            reliability_data = self.extract_reliability_feature(train_df, train_links)
            post_data = self.encode_post(train_df)
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])
            self.train_dataset = TensorDataset(post_data, simple_wiki_data, reliability_data, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                 delimiter='\t')
            val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']
            simple_wiki_data = self.extract_simple_wiki_feature(val_df, val_links)
            reliability_data = self.extract_reliability_feature(val_df, val_links)
            post_data = self.encode_post(val_df)
            val_labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in val_labels])
            self.val_dataset = TensorDataset(post_data, simple_wiki_data, reliability_data, labels)

        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                  delimiter='\t')
            test_links = json.load(open(self.hparams.link_test_path, 'r'))['links']
            simple_wiki_data = self.extract_simple_wiki_feature(test_df, test_links)
            reliability_data = self.extract_reliability_feature(test_df, test_links)
            post_data = self.encode_post(test_df)
            self.test_dataset = TensorDataset(post_data, simple_wiki_data, reliability_data)


class HistoryLinks(Links):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if self.hparams.train_path:
            train_df = pd.read_csv(self.hparams.train_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                   delimiter='\t')
            train_links = json.load(open(self.hparams.link_train_path, 'r'))['links']
            train_history_df = pd.read_csv(self.hparams.history_train_path, sep='\t')
            simple_wiki_data = self.extract_simple_wiki_feature(train_df, train_links)
            reliability_data = self.extract_reliability_feature(train_df, train_links)
            history = self.encode_history(train_df, train_history_df)
            post = self.encode_post(train_df)
            train_labels = train_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in train_labels])
            self.train_dataset = TensorDataset(history, post, simple_wiki_data, reliability_data, labels)

        if self.hparams.val_path:
            val_df = pd.read_csv(self.hparams.val_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                 delimiter='\t')
            val_links = json.load(open(self.hparams.link_val_path, 'r'))['links']
            val_history_df = pd.read_csv(self.hparams.history_val_path, sep='\t')
            simple_wiki_data = self.extract_simple_wiki_feature(val_df, val_links)
            reliability_data = self.extract_reliability_feature(val_df, val_links)
            history = self.encode_history(val_df, val_history_df)
            val_labels = val_df.label.tolist()
            labels = torch.tensor([self.labels2id[i] for i in val_labels])
            post = self.encode_post(val_df)
            self.val_dataset = TensorDataset(history, post,  simple_wiki_data, reliability_data, labels)

        if self.hparams.test_path:
            test_df = pd.read_csv(self.hparams.test_path, quoting=csv.QUOTE_NONE, error_bad_lines=False,
                                  delimiter='\t')
            test_links = json.load(open(self.hparams.link_test_path, 'r'))['links']
            test_history_df = pd.read_csv(self.hparams.history_test_path, sep='\t')
            simple_wiki_data = self.extract_simple_wiki_feature(test_df, test_links)
            reliability_data = self.extract_reliability_feature(test_df, test_links)
            history = self.encode_history(test_df, test_history_df)
            post = self.encode_post(test_df)
            self.test_dataset = TensorDataset(history,post, simple_wiki_data, reliability_data)


DATA_MODELS = {
    'style': Style,
    'history': History,
    'links': Links,
    'history_style': HistoryStyle,
    'history_links': HistoryLinks,
    'history_links_style': HistoryStyleLinks,
    'links_style': LinksStyle
}
