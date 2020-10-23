## Start : Import the packages
import pandas as pd
import os
import pathlib
import zipfile
# import wget
# import gdown
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src.logger import logger
from cleantext import clean

def clean_helper(self, text):
    return clean(text,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 no_urls=True,  # replace all URLs with a special token
                 no_emails=True,
                 lower=True,
                 # replace all email addresses with a special token
                 no_phone_numbers=True,
                 # replace all phone numbers with a special token
                 no_numbers=True,  # replace all numbers with a special token
                 no_digits=True,  # replace all digits with a special token
                 no_currency_symbols=True,
                 # replace all currency symbols with a special token
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="<DIGIT>",
                 replace_with_currency_symbol="<CUR>",
                 lang="en")

# Acknowledgement: The original version of the following code is https://arthought.com/transformer-model-fine-tuning-for-text-classification-with-pytorch-lightning/
# I have modified a bit.

# End : Import the packages
# %%
# Remark: pl.LightningModule is derived from torch.nn.Module It has additional methods that are part of the
# lightning interface and that need to be defined by the user. Having these additional methods is very useful
# for several reasons:
# 1. Reasonable Expectations: Once you know the pytorch-lightning-system you more easily read other people's
#    code, because it is always structured in the same way
# 2. Less Boilerplate: The additional class methods make pl.LightningModule more powerful than the nn.Module
#    from plain pytorch. This means you have to write less of the repetitive boilerplate code
# 3. Perfact for the development lifecycle Pytorch Lightning makes it very easy to switch from cpu to gpu/tpu.
#    Further it supplies method to quickly run your code on a fraction of the data, which is very useful in
#    the development process, especially for debugging


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        # freeze
        self._frozen = False
        config = AutoConfig.from_pretrained(self.hparams.pretrained,
                                            num_labels=self.hparams.num_labels,
                                            output_attentions=False,
                                            output_hidden_states=False)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained, config=config)

    def forward(self, batch):
        b_input_ids = batch[0]
        b_input_mask = batch[1]

        has_labels = len(batch) > 2

        b_labels = batch[2] if has_labels else None
        if has_labels:
            loss, logits = self.model(b_input_ids,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
        if not has_labels:
            loss, logits = None, self.model(b_input_ids,
                                            attention_mask=b_input_mask,
                                            labels=b_labels)

        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self(
            batch
        )  # self refers to the model, which in turn acceses the forward method

        tensorboard_logs = {'train_loss': loss}
        # pytorch lightning allows you to use various logging facilities, eg tensorboard with tensorboard we
        # can track and easily visualise the progress of training. In this case

        return {'loss': loss, 'log': tensorboard_logs}
        # the training_step method expects a dictionary, which should at least contain the loss

    def validation_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do validation. This we do here, by virtue of this
        # definition.

        loss, logits = self(batch)
        # self refers to the model, which in turn accesses the forward method

        # Apart from the validation loss, we also want to track validation accuracy  to get an idea, what the
        # model training has achieved "in real terms".

        labels = batch[2]
        predictions = torch.argmax(logits, dim=1)
        accuracy = (labels == predictions).float().mean()

        return {'val_loss': loss, 'accuracy': accuracy}
        # the validation_step method expects a dictionary, which should at least contain the val_loss

    def validation_epoch_end(self, validation_step_outputs):
        # OPTIONAL The second parameter in the validation_epoch_end - we named it validation_step_outputs -
        # contains the outputs of the validation_step, collected for all the batches over the entire epoch.

        # We use it to track progress of the entire epoch, by calculating averages

        avg_loss = torch.stack([x['val_loss']
                                for x in validation_step_outputs]).mean()

        avg_accuracy = torch.stack(
            [x['accuracy'] for x in validation_step_outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_accuracy}
        return {
            'val_loss': avg_loss,
            'log': tensorboard_logs,
            'progress_bar': {
                'avg_loss': avg_loss,
                'avg_accuracy': avg_accuracy
            }
        }
        # The training_step method expects a dictionary, which should at least contain the val_loss. We also
        # use it to include the log - with the tensorboard logs. Further we define some values that are
        # displayed in the tqdm-based progress bar.

    def configure_optimizers(self):
        # The configure_optimizers is a (virtual) method, specified in the interface, that the
        # pl.LightningModule class wants you to overwrite.

        # In this case we define that some parameters are optimized in a different way than others. In
        # particular we single out parameters that have 'bias', 'LayerNorm.weight' in their names. For those
        # we do not use an optimization technique called weight decay.

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                0.01
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=1e-8
                          # args.adam_epsilon  - default is 1e-8.
                          )

        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            # Default value in run_glue.py
            num_training_steps=self.hparams.num_training_steps)

        return [optimizer], [scheduler]

    def freeze(self) -> None:
        # freeze all layers, except the final classifier layers
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

        self._frozen = True

    def unfreeze(self) -> None:
        if self._frozen:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = True

        self._frozen = False

    def on_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()


class ConstraintData(pl.LightningDataModule):
    # So here we finally arrive at the definition of our data class derived from pl.LightningDataModule.
    #
    # In earlier versions of pytorch lightning  (prior to 0.9) the methods here were part of the model class
    # derived from pl.LightningModule. For better flexibility and readability the Data and Model related parts
    # were split out into two different classes:
    #
    # pl.LightningDataModule and pl.LightningModule
    #
    # with the Model related part remaining in pl.LightningModule
    #
    # This is explained in more detail in this video: https://www.youtube.com/watch?v=L---MBeSXFw

    def __init__(self, *args, **kwargs):
        super().__init__()

        # self.save_hyperparameters()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        # cf this open issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/3232

        print('args:', args)
        print('kwargs:', kwargs)

        # print(f'self.hparams.pretrained:{self.hparams.pretrained}')

        print('Loading BERT tokenizer')
        print(f'PRETRAINED:{self.hparams.pretrained}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained)

        print('Type tokenizer:', type(self.tokenizer))

        # This part is easy  we instantiate the tokenizer

        # So this is easy, but it's also incredibly important, e.g. in this by using "bert-base-uncased", we
        # determine, that before any text is analysed its all turned into lower case. This might have a
        # significant impact on model performance!!!
        #
        # BertTokenizer is the tokenizer, which is loaded automatically by AutoTokenizer because it was used
        # to train the model weights of "bert-base-uncased".

        self.labels2id = {'fake': 0, 'true': 1}
        self.id2labels = {val: key for key, val in self.labels2id.items()}

    def prepare_data(self):
        # Even if you have a complicated setup, where you train on a cluster of multiple GPUs, prepare_data is
        # only run once on the cluster.

        # Typically - as done here - prepare_data just performs the time-consuming step of downloading the
        # data.
        pass

    def setup(self, stage=None):
        # Even if you have a complicated setup, where you train on a cluster of multiple GPUs, setup is run
        # once on every gpu of the cluster.

        # typically - as done here - setup
        # - reads the previously downloaded data
        # - does some preprocessing such as tokenization

        # Load the dataset into a pandas dataframe.
        train_df = pd.read_csv('./data/train.tsv', delimiter='\t')
        val_df = pd.read_csv('./data/val.tsv', delimiter='\t')

        # Stats of dataset
        logger.info(f'Total samples in training: {len(train_df)}')
        logger.info(f'Total samples in validation: {len(val_df)}')

        # Get the lists of sentences and their labels.
        train_tweets = train_df.tweet.tolist()
        train_labels = train_df.label.tolist()

        # tokenize the sentences with Transformer tokens
        train_encoded_tweets = self.tokenizer(
            train_tweets,  # Sentence to encode.
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
        train_labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # Combine the training inputs into a TensorDataset.
        self.train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

        val_tweets = val_df.tweet.tolist()
        val_labels = val_df.label.tolist()

        # tokenize the sentences with Transformer tokens
        val_encoded_tweets = self.tokenizer(
            val_tweets,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.hparams.max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )

        # Convert the lists into tensors.
        val_input_ids = val_encoded_tweets['input_ids']
        val_attention_mask = train_encoded_tweets['attention_mask']
        val_labels = torch.tensor([self.labels2id[i] for i in val_labels])

        # Combine the training inputs into a TensorDataset.
        self.val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

    def train_dataloader(self):
        # as explained above, train_dataloader was previously part of the model class derived from
        # pl.LightningModule train_dataloader needs to return the a Dataloader with the train_dataset

        return DataLoader(
            self.train_dataset,  # The training samples.
            sampler=RandomSampler(
                self.train_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size  # Trains with this batch size.
        )

    def val_dataloader(self):
        # as explained above, train_dataloader was previously part of the model class derived from
        # pl.LightningModule train_dataloader needs to return the a Dataloader with the val_dataset

        return DataLoader(
            self.val_dataset,  # The training samples.
            sampler=RandomSampler(self.val_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size,  # Trains with this batch size.
            shuffle=False)


class NELAData(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # self.save_hyperparameters()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        # cf this open issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/3232

        print('args:', args)
        print('kwargs:', kwargs)

        # print(f'self.hparams.pretrained:{self.hparams.pretrained}')

        print('Loading BERT tokenizer')
        print(f'PRETRAINED:{self.hparams.pretrained}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained)

        print('Type tokenizer:', type(self.tokenizer))

        self.labels2id = {'reliable': 0, 'unreliable': 1, 'satire': 2}
        self.id2labels = {val: key for key, val in self.labels2id.items()}

    def prepare_data(self):
        # Even if you have a complicated setup, where you train on a cluster of multiple GPUs, prepare_data is
        # only run once on the cluster.

        # Typically - as done here - prepare_data just performs the time-consuming step of downloading the
        # data.
        pass

    def setup(self, stage=None):
        # Even if you have a complicated setup, where you train on a cluster of multiple GPUs, setup is run
        # once on every gpu of the cluster.

        # typically - as done here - setup
        # - reads the previously downloaded data
        # - does some preprocessing such as tokenization

        # Load the dataset into a pandas dataframe.
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
            train_title,train_content,  # Sentence to encode.
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
        train_labels = torch.tensor([self.labels2id[i] for i in train_labels])

        # Combine the training inputs into a TensorDataset.
        self.train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

        val_tweets = val_df.tweet.tolist()
        val_labels = val_df.label.tolist()

        # tokenize the sentences with Transformer tokens
        val_encoded_tweets = self.tokenizer(
            val_tweets,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.hparams.max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )

        # Convert the lists into tensors.
        val_input_ids = val_encoded_tweets['input_ids']
        val_attention_mask = train_encoded_tweets['attention_mask']
        val_labels = torch.tensor([self.labels2id[i] for i in val_labels])

        # Combine the training inputs into a TensorDataset.
        self.val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

    def train_dataloader(self):
        # as explained above, train_dataloader was previously part of the model class derived from
        # pl.LightningModule train_dataloader needs to return the a Dataloader with the train_dataset

        return DataLoader(
            self.train_dataset,  # The training samples.
            sampler=RandomSampler(
                self.train_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size  # Trains with this batch size.
        )

    def val_dataloader(self):
        # as explained above, train_dataloader was previously part of the model class derived from
        # pl.LightningModule train_dataloader needs to return the a Dataloader with the val_dataset

        return DataLoader(
            self.val_dataset,  # The training samples.
            sampler=RandomSampler(self.val_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size,  # Trains with this batch size.
            shuffle=False)


# %%
if __name__ == "__main__":
    # Two key aspects:

    # - pytorch lightning can add arguments to the parser automatically

    # - you can manually add your own specific arguments.

    # - there is a little more code than seems necessary, because of a particular argument the scheduler
    #   needs. There is currently an open issue on this complication
    #   https://github.com/PyTorchLightning/pytorch-lightning/issues/1038

    import argparse
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # We use the parts of very convenient Auto functions from huggingface. This way we can easily switch
    # between models and tokenizers, just by giving a different name of the pretrained model.
    #
    # BertForSequenceClassification is one of those pretrained models, which is loaded automatically by
    # AutoModelForSequenceClassification because it corresponds to the pretrained weights of
    # "bert-base-uncased".

    # Similarly BertTokenizer is one of those tokenizers, which is loaded automatically by AutoTokenizer
    # because it is the necessary tokenizer for the pretrained weights of "bert-base-uncased".
    parser.add_argument('--pretrained', type=str, default="bert-base-uncased")
    parser.add_argument('--batch_size', type=float, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--nela_train', type=str, default='NELA/train_1.tsv')
    parser.add_argument('--nela_test', type=str, default='NELA/test_1.tsv')

    # parser = Model.add_model_specific_args(parser) parser = Data.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # TODO start: remove this later
    # args.limit_train_batches = 10 # TODO remove this later
    # args.limit_val_batches = 5 # TODO remove this later
    # args.frac = 0.01 # TODO remove this later
    # TODO end: remove this later

    # start : get training steps
    d = NELAData(args)
    d.prepare_data()
    d.setup()
    args.num_training_steps = len(d.train_dataloader()) * args.max_epochs
    # end : get training steps

    dict_args = vars(args)
    m = Model(**dict_args)

    args.early_stop_callback = EarlyStopping('val_loss')

    trainer = pl.Trainer.from_argparse_args(args)

    # fit the data
    trainer.fit(m, d)

    # %%
