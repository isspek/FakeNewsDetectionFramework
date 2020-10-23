import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
from src.logger import logger
from cleantext import clean


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


class ConstraintData(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained)
        self.labels2id = {'fake': 0, 'true': 1}
        self.id2labels = {val: key for key, val in self.labels2id.items()}

    def setup(self, stage=None):
        train_df = pd.read_csv(self.hparams.train_path, delimiter='\t')
        val_df = pd.read_csv(self.hparams.val_path, delimiter='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        # Get the lists of sentences and their labels.
        train_tweets = train_df.tweet.map(lambda x: clean_helper(x)).tolist()
        train_labels = train_df.label.tolist()

        self.train_dataset = TensorDataset(self.encode_for_transformer(train_labels, train_tweets))

        val_tweets = val_df.tweet.map(lambda x: clean_helper(x)).tolist()
        val_labels = val_df.label.tolist()

        self.val_dataset = TensorDataset(self.encode_for_transformer(val_tweets, val_labels))

    def encode_for_transformer(self, tweets, labels):
        encoded_tweets = self.tokenizer(
            tweets,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.hparams.max_len,  # Pad & truncate all sentences.
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
            batch_size=self.hparams.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.hparams.batch_size,
            shuffle=False)


class NELAData(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained)
        self.labels2id = {'reliable': 0, 'unreliable': 1, 'satire': 2}
        self.id2labels = {val: key for key, val in self.labels2id.items()}

    def setup(self, stage=None):
        train_df = pd.read_csv(self.hparams.nela_train, delimiter='\t')
        val_df = pd.read_csv(self.hparams.nela_test, delimiter='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        # Get the lists of sentences and their labels.
        train_title = train_df.title.map(lambda x: clean_helper(x)).tolist()
        train_content = train_df.content.map(lambda x: clean_helper(x)).tolist()
        train_labels = train_df.label.tolist()
        train_labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # tokenize the sentences with Transformer tokens
        train_encoded_tweets = self.tokenizer(
            train_title, train_content,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.hparams.max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )

        # Convert the lists into tensors.
        train_input_ids = train_encoded_tweets['input_ids']
        train_attention_mask = train_encoded_tweets['attention_mask']

        # Combine the training inputs into a TensorDataset.
        self.train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        val_title = val_df.title.map(lambda x: clean_helper(x)).tolist()
        val_content = val_df.content.map(lambda x: clean_helper(x)).tolist()
        val_labels = val_df.label.tolist()
        val_labels = torch.tensor([self.labels2id[i] for i in val_labels])

        # tokenize the sentences with Transformer tokens
        val_encoded_tweets = self.tokenizer(
            val_title, val_content,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.hparams.max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )
        val_input_ids = val_encoded_tweets['input_ids']
        val_attention_mask = val_encoded_tweets['attention_mask']
        self.val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # The training samples.
            sampler=RandomSampler(
                self.train_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size,  # Trains with this batch size.
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # The training samples.
            sampler=SequentialSampler(self.val_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size,  # Trains with this batch size.
            shuffle=False)


DATA_MODELS = {
    'constraint': ConstraintData,
    'nela': NELAData
}
