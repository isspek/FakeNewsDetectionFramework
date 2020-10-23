import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

BERT_MODEL = 'bert-base-uncased'
config = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3
}


class TransformerTuner(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(self.hparams.model_name, num_labels=self.hparams.num_labels,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name, config=self.config)

    def forward(self, batch):
        input_ids = batch[0]
        input_mask = batch[1]

        has_labels = batch[2] if len(batch) > 2 else None




if __name__ == '__main__':
    pass
