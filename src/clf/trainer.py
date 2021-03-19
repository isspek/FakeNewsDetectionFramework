import torch
import os
import time
import glob
import numpy as np
from transformers import AutoModel
from transformers import AutoConfig
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import pytorch_lightning as pl
import argparse
from argparse import ArgumentParser
from .data_models import DATA_MODELS, NUM_OF_PAST_URLS, NUM_OF_PAST_CLAIMS
from pathlib import Path
from transformers import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Any, Dict
from pytorch_lightning.utilities import rank_zero_info
from transformers.data.metrics import simple_accuracy
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from torch.nn.functional import softmax
import torch.nn as nn
from tqdm import tqdm


class Constraint(pl.LightningModule):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading

        self.save_hyperparameters(hparams)
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": self.hparams.num_labels} if self.hparams.num_labels is not None else {}),
                cache_dir=self.cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        self.model_type = AutoModel
        self.config.output_attentions = True
        if model is None:
            self.transformer_model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=self.cache_dir
            )
        else:
            self.transformer_model = model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.transformer_model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.transformer_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def _eval_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        preds = np.argmax(preds, axis=1)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **{"acc": simple_accuracy(preds, out_label_ids)}}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.hparams.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.transformer_model.config.save_step = self.step_count
        self.transformer_model.save_pretrained(save_path)

    def encode_post(self, post):
        post = self.transformer_model(post[:, 0, :, :].squeeze(dim=1), token_type_ids=None,
                                      attention_mask=post[:, 1, :, :].squeeze(dim=1))[1]
        post = self.dropout(post)
        return post

    def encode_reliability(self, reliability):
        reliability = reliability.contiguous().view(reliability.shape[0], -1)
        return reliability

    def encode_topics(self, topics):
        topics = topics.contiguous().view(topics.shape[0], -1)
        return topics

    def encode_history(self, past_claims):
        past_claims_len = past_claims.shape[1]
        past_claims_embedding = []
        for i in range(past_claims_len):
            input_ids = past_claims[:, i, 0, :, :]
            attention_masks = past_claims[:, i, 1, :, :]
            pooled_output = self.transformer_model(input_ids.squeeze(dim=1), token_type_ids=None,
                                                   attention_mask=attention_masks.squeeze(dim=1))[1]
            pooled_output = self.dropout(pooled_output)
            past_claims_embedding.append(pooled_output)
        past_claims_embedding = torch.sum(torch.stack(past_claims_embedding), dim=0)
        past_claims_embedding = torch.div(past_claims_embedding, past_claims_len)
        past_claims_embedding = self.dropout(past_claims_embedding)
        return past_claims_embedding

    def encode_wiki(self, simple_wiki):
        simple_wiki_len = simple_wiki.shape[1]
        simple_wiki_embedding = []
        for i in range(simple_wiki_len):
            input_ids = simple_wiki[:, i, 0, :, :]
            attention_masks = simple_wiki[:, i, 1, :, :]
            pooled_output = self.transformer_model(input_ids.squeeze(dim=1), token_type_ids=None,
                                                   attention_mask=attention_masks.squeeze(dim=1))[1]
            simple_wiki_embedding.append(pooled_output)
        simple_wiki_embedding = torch.cat(simple_wiki_embedding, dim=1)
        simple_wiki_embedding = self.dropout(simple_wiki_embedding)
        return simple_wiki_embedding


class History(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.classifier = nn.Linear(1, self.num_labels)
        self.classifier = nn.Linear(1, self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {"past_claims": batch[0], "post": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        past_claims = inputs['past_claims']
        post = inputs['post']
        labels = inputs['labels'] if 'labels' in inputs else None
        post = self.encode_post(post)
        past_claims = self.encode_history(past_claims)
        sim_score = self.cos(past_claims, post).unsqueeze(dim=0)
        logits = self.classifier(sim_score)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        inputs = {"past_claims": batch[0], "post": batch[1], "labels": batch[2]}
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu() if tmp_eval_loss else None, "pred": preds,
                "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {"past_claims": batch[0], "post": batch[1]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class HistoryStyle(History):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.classifier = nn.Linear(self.config.hidden_size + 1,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'labels': batch[2]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)

        past_claims = inputs['past_claims']
        past_claims_embedding = self.encode_history(past_claims)
        sim_score = self.cos(past_claims_embedding, post).unsqueeze(dim=0)

        auxiliary_list = []
        cos_sims = []
        cos_sims.append(sim_score)
        auxiliary_list.append(post)
        auxiliary_list.append(sim_score)
        auxiliary_list = torch.cat(auxiliary_list, dim=1)

        logits = self.classifier(auxiliary_list)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, auxiliary_list, cos_sims

    def validation_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'labels': batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}

class HistoryStyleV2(History):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.classifier = nn.Linear(self.config.hidden_size + self.config.hidden_size,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'labels': batch[2]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)

        past_claims = inputs['past_claims']
        past_claims_embedding = self.encode_history(past_claims)
        # sim_score = self.cos(past_claims_embedding, post).unsqueeze(dim=0)

        auxiliary_list = []
        auxiliary_list.append(post)
        auxiliary_list.append(past_claims_embedding)
        auxiliary_list = torch.cat(auxiliary_list, dim=1)

        logits = self.classifier(auxiliary_list)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'labels': batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class Links(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.wiki = self.hparams.wiki
        self.reliability = self.hparams.reliability
        if self.wiki:
            self.classifier = nn.Linear(self.config.hidden_size * NUM_OF_PAST_URLS,
                                        self.num_labels)
        elif self.reliability:
            self.classifier = nn.Linear(20,
                                        self.num_labels)  # 5 comes from reliability encoders
        else:
            self.classifier = nn.Linear(self.config.hidden_size * NUM_OF_PAST_URLS + 20,
                                        self.num_labels)  # 5 comes from reliability encoders

    def training_step(self, batch, batch_idx):
        if self.wiki:
            inputs = {'simple_wiki': batch[0], "labels": batch[1]}
        elif self.reliability:
            inputs = {"reliability": batch[0], "labels": batch[1]}
        else:
            inputs = {'simple_wiki': batch[0], 'reliability': batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):

        labels = inputs['labels'] if 'labels' in inputs else None
        auxiliary_list = []
        if self.wiki:
            simple_wiki = inputs['simple_wiki']
            concat_embeddings = []
            simple_wiki_len = simple_wiki.shape[1]
            for i in range(simple_wiki_len):
                input_ids = simple_wiki[:, i, 0, :, :]
                attention_masks = simple_wiki[:, i, 1, :, :]
                pooled_output = self.transformer_model(input_ids.squeeze(dim=1), token_type_ids=None,
                                                       attention_mask=attention_masks.squeeze(dim=1))[1]
                concat_embeddings.append(pooled_output)

            concat_embeddings = torch.cat(concat_embeddings, dim=1)
            concat_embeddings = self.dropout(concat_embeddings)
            auxiliary_list.append(concat_embeddings)
        elif self.reliability:
            reliability = inputs['reliability']
            reliability = reliability.contiguous().view(reliability.shape[0], -1)
            auxiliary_list.append(reliability)
        else:
            simple_wiki = inputs['simple_wiki']
            concat_embeddings = []
            simple_wiki_len = simple_wiki.shape[1]
            for i in range(simple_wiki_len):
                input_ids = simple_wiki[:, i, 0, :, :]
                attention_masks = simple_wiki[:, i, 1, :, :]
                pooled_output = self.transformer_model(input_ids.squeeze(dim=1), token_type_ids=None,
                                                       attention_mask=attention_masks.squeeze(dim=1))[1]
                concat_embeddings.append(pooled_output)

            concat_embeddings = torch.cat(concat_embeddings, dim=1)
            concat_embeddings = self.dropout(concat_embeddings)
            auxiliary_list.append(concat_embeddings)
            reliability = inputs['reliability']
            reliability = reliability.contiguous().view(reliability.shape[0], -1)
            auxiliary_list.append(reliability)

        auxiliary_list = torch.cat(auxiliary_list, dim=1)
        logits = self.classifier(auxiliary_list.float())
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        if self.wiki:
            inputs = {'simple_wiki': batch[0], "labels": batch[1]}
        elif self.reliability:
            inputs = {"reliability": batch[0], "labels": batch[1]}
        else:
            inputs = {'simple_wiki': batch[0], 'reliability': batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        if self.wiki:
            inputs = {'simple_wiki': batch[0]}
        elif self.reliability:
            inputs = {"reliability": batch[0]}
        else:
            inputs = {'simple_wiki': batch[0], 'reliability': batch[1]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class Style(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'labels': batch[1]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)
        logits = self.classifier(post)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, post

    def validation_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'labels': batch[1]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'post': batch[0]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class HistoryLinksStyle(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.classifier = nn.Linear(self.config.hidden_size * (NUM_OF_PAST_URLS + 1) + 21,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'simple_wiki': batch[2], 'reliability': batch[3],
                  'labels': batch[4]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)

        simple_wiki = inputs['simple_wiki']
        simple_wiki_embedding = self.encode_wiki(simple_wiki)

        past_claims = inputs['past_claims']
        past_claims_embedding = self.encode_history(past_claims)
        sim_score = self.cos(past_claims_embedding, post).unsqueeze(dim=0)

        reliability = inputs['reliability']
        reliability = self.encode_reliability(reliability)

        auxiliary_list = []
        auxiliary_list.append(simple_wiki_embedding)
        auxiliary_list.append(post)
        auxiliary_list.append(sim_score)
        auxiliary_list.append(reliability)
        auxiliary_list = torch.cat(auxiliary_list, dim=1)

        logits = self.classifier(auxiliary_list)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'simple_wiki': batch[2], 'reliability': batch[3],
                  'labels': batch[4]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'simple_wiki': batch[2], 'reliability': batch[3]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class LinksStyle(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size * (NUM_OF_PAST_URLS + 1) + 20,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'simple_wiki': batch[1], 'reliability': batch[2],
                  'labels': batch[3]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)

        simple_wiki = inputs['simple_wiki']
        simple_wiki_embedding = self.encode_wiki(simple_wiki)

        reliability = inputs['reliability']
        reliability = self.encode_reliability(reliability)

        auxiliary_list = []
        auxiliary_list.append(simple_wiki_embedding)
        auxiliary_list.append(post)
        auxiliary_list.append(reliability)
        auxiliary_list = torch.cat(auxiliary_list, dim=1)

        logits = self.classifier(auxiliary_list)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'simple_wiki': batch[1], 'reliability': batch[2],
                  'labels': batch[3]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'simple_wiki': batch[1], 'reliability': batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class TopicStyle(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size + 20,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'topics': batch[1],
                  'labels': batch[2]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)

        topics = inputs['topics']
        topics = self.encode_topics(topics)

        auxiliary_list = []
        auxiliary_list.append(post)
        auxiliary_list.append(topics)
        auxiliary_list = torch.cat(auxiliary_list, dim=1)

        logits = self.classifier(auxiliary_list)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, auxiliary_list

    def validation_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'topics': batch[1],
                  'labels': batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'post': batch[0],
                  'topics': batch[1]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class HistoryLinks(Constraint):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            model=None,
            **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__(hparams, config=config, model=model, **config_kwargs)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.classifier = nn.Linear(self.config.hidden_size * (NUM_OF_PAST_URLS) + 21,
                                    self.num_labels)

    def training_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'simple_wiki': batch[2], 'reliability': batch[3],
                  'labels': batch[4]}

        outputs = self(**inputs)
        loss = outputs[0]
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def forward(self, **inputs):
        labels = inputs['labels'] if 'labels' in inputs else None

        post = inputs['post']
        post = self.encode_post(post)

        simple_wiki = inputs['simple_wiki']
        simple_wiki_embedding = self.encode_wiki(simple_wiki)

        past_claims = inputs['past_claims']
        past_claims_embedding = self.encode_history(past_claims)
        sim_score = self.cos(past_claims_embedding, post).unsqueeze(dim=0)

        reliability = inputs['reliability']
        reliability = self.encode_reliability(reliability)

        auxiliary_list = []
        auxiliary_list.append(simple_wiki_embedding)
        auxiliary_list.append(sim_score)
        auxiliary_list.append(reliability)
        auxiliary_list = torch.cat(auxiliary_list, dim=1)

        logits = self.classifier(auxiliary_list)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'simple_wiki': batch[2], 'reliability': batch[3],
                  'labels': batch[4]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_idx):
        inputs = {'past_claims': batch[0], 'post': batch[1],
                  'simple_wiki': batch[2], 'reliability': batch[3]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()

        return {"pred": preds}


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir) -> None:
    #  To allow all pl args uncomment the following line
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--task",
        default="",
        type=str,
        choices=['history', 'links', 'history_style', 'history_links', 'links_style', 'history_links_style',
                 'history_links_nowiki', 'links_nowiki', 'history_links_style_nowiki', 'history_links_style_onlywiki',
                 'style', 'topic_style'],
        help="Fakenews tasks",
    )

    parser.add_argument(
        "--gpus",
        default=0,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--train_path", type=str, help="path for train set")
    parser.add_argument("--val_path", type=str, help="path for dev set")
    parser.add_argument("--test_path", type=str, help="path for test set")
    parser.add_argument("--history_train_path", type=str, help="path for search results of train set")
    parser.add_argument("--history_val_path", type=str, help="path for search results of val set")
    parser.add_argument("--history_test_path", type=str, help="path for search results of val set")
    parser.add_argument("--link_train_path", type=str, help="path for links of train set")
    parser.add_argument("--link_val_path", type=str, help="path for links of val set")
    parser.add_argument("--link_test_path", type=str, help="path for links of val set")
    parser.add_argument("--output_fname", type=str, default='test_results.csv')
    parser.add_argument("--col_name", type=str,
                        help='column name of the dataset which presents textual field e.g content or tweet')
    parser.add_argument("--data", type=str,
                        help='dataset name, it might be required to handle errors', default='na')

    parser.add_argument(
        "--num_labels",
        default=2,
        type=int,
        help="Add number of class for the classification",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--encoder_layerdrop",
        type=float,
        help="Encoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--decoder_layerdrop",
        type=float,
        help="Decoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        help="Attention dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        type=str,
        help="Learning rate scheduler",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--wiki", action="store_true")
    parser.add_argument("--reliability", action="store_true")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument("--output_attention", action="store_true")

    return parser


def generic_train(
        model: History,
        data: DataLoader,
        args: argparse.Namespace,
        logger=True,  # can pass WandbLogger() here
        extra_callbacks=[],
        checkpoint_callback=None,
        logging_callback=None,
        **extra_train_kwargs
):
    pl.seed_everything(args.seed)

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model, data)

    return trainer


# reference https://github.com/ricardorei/lightning-text-classification
def mask_fill(
        fill_value: float,
        tokens: torch.tensor,
        embeddings: torch.tensor,
        padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


MODELS = {
    'style': Style,
    'history': History,
    'links': Links,
    'topic_style': TopicStyle,
    'history_style': HistoryStyle,
    'history_style_v2': HistoryStyleV2,
    'history_links': HistoryLinks,
    'history_links_style': HistoryLinksStyle,
    'links_style': LinksStyle,
}

# %%
if __name__ == "__main__":
    parser = ArgumentParser()
    add_generic_args(parser, os.getcwd())

    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    # start : get training steps
    data = DATA_MODELS[args.task](args)
    data.prepare_data()
    data.setup()

    if args.do_train:
        args.dataset_size = len(data.train_dataloader())

    model = MODELS[args.task](args)

    # train model
    trainer = generic_train(model, data, args)

    loader = data.test_dataloader()
    # Optionally, predict on dev set and write to output_dir
    # if args.do_predict:
    # loader = data.val_dataloader()
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"), recursive=True)))
    model = model.load_from_checkpoint(checkpoints[-1])

    # inference
    preds = []
    probs = []

    device = torch.device('cuda')
    model.to(device)
    model.eval()
    for ix, batch in enumerate(loader):
        if args.task == 'history':
            # inputs = {"past_claims": batch[0].to(device)}
            inputs = {"past_claims": batch[0].to(device), "post": batch[1].to(device)}
        elif args.task == 'history_style' or args.task == 'history_style_v2':
            inputs = {"past_claims": batch[0].to(device), "post": batch[1].to(device)}
        elif args.task == 'links':
            if args.wiki:
                inputs = {'simple_wiki': batch[0].to(device)}
                # inputs = {'simple_wiki': batch[0].to(device), 'reliability': batch[1].to(device)}
            elif args.reliability:
                inputs = {'reliability': batch[0].to(device)}
            else:
                inputs = {'simple_wiki': batch[0].to(device), 'reliability': batch[1].to(device)}
        elif args.task == 'topic_style':
            inputs = {'post': batch[0].to(device), 'topics': batch[1].to(device)}
        elif args.task == 'history_links':
            inputs = {"past_claims": batch[0].to(device), 'post': batch[1].to(device),
                      'simple_wiki': batch[2].to(device),
                      'reliability': batch[3].to(device)}
        elif args.task == 'links_style':
            inputs = {'post': batch[0].to(device), 'simple_wiki': batch[1].to(device),
                      'reliability': batch[2].to(device)}
        elif args.task == 'history_links_style':
            inputs = {'past_claims': batch[0].to(device), 'post': batch[1].to(device),
                      'simple_wiki': batch[2].to(device), 'reliability': batch[3].to(device)}
        elif args.task == 'style':
            inputs = {"post": batch[0].to(device)}
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            prob = softmax(logits, dim=1)
            probs.append(torch.max(prob).cpu().detach().numpy().item())
            preds.append(int(torch.argmax(logits).cpu().numpy().item()))
        # labels.append(inputs["labels"].cpu().numpy().item())

    preds = [data.id2labels[pred] for pred in preds]
    # labels = [data.id2labels[label] for label in labels]

    validation = pd.DataFrame({
        'predictions': preds,
        'probs': probs,
    })
    validation.to_csv(os.path.join(args.output_dir, args.output_fname))
