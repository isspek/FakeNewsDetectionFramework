import torch
import pandas as pd
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments, set_seed
from torch.utils.data import Dataset
from cleantext import clean
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from ..logger import logger
import numpy as np
from torch import nn


class NELADataset(Dataset):
    def __init__(self, path, tokenizer):
        # self.str2id = {'reliable': 0, 'mixed': 1, 'unreliable': 2, 'satire': 3}
        self.str2id = {'reliable': 0, 'unreliable': 1, 'satire': 2}
        self.id2str = {val: key for key, val in self.str2id.items()}
        self.num_labels = len(self.str2id.keys())
        data = pd.read_csv(path, sep='\t')
        self.size = len(data)
        data['title'] = data.title.map(lambda x: self.clean_helper(x))
        data['content'] = data.content.map(lambda x: self.clean_helper(x))
        data['label'] = data.label.map(lambda x: self.str2id[x])
        title, content = data['title'].tolist(), data['content'].tolist()
        self.encodings = tokenizer(title, content, padding=True, truncation=True, verbose=False,
                                   return_tensors='pt', max_length=512)

        self.labels = data['label'].tolist()

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

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def _clean(self, text):
        return

    def __len__(self):
        return self.size


class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, train):
        pass

    @abstractmethod
    def eval(self, test, model_path):
        pass


class Transformer(Model):
    def __init__(self, output_dir, model_name):
        self.output_dir = output_dir
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self, config, train_dir):
        train_dataset = NELADataset(train_dir, self.tokenizer)
        logger.info(f'Training samples: {train_dataset.__len__()}')
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=train_dataset.num_labels)
        temp_dir = Path(self.output_dir) / 'temp'
        training_args = TrainingArguments(
            output_dir=temp_dir,  # output directory
            overwrite_output_dir=True,
            do_train=True,
            # do_eval=True,
            learning_rate=config['learning_rate'],
            num_train_epochs=config['num_train_epochs'],  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=1,  # batch size for evaluation
            logging_dir='./logs',  # directory for storing logs
            logging_steps=500,
            load_best_model_at_end=True,
            save_steps=500,
            save_total_limit=1
        )

        set_seed(training_args.seed)

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
        )

        shutil.rmtree(temp_dir)
        trainer.save_model(self.output_dir)

    def eval(self, test_dir, model_path):
        logger.info('Evaluating...')
        test_dataset = NELADataset(test_dir, self.tokenizer)

        device = torch.device('cuda')
        config = AutoConfig.from_pretrained(model_path, num_labels=test_dataset.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        test_iter = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)

        model.to(device)
        model.eval()
        test_y = []
        preds = []

        for batch in tqdm(test_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # forward pass
            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     labels=labels)

            # record preds, trues
            _pred = logits.cpu().data.numpy()
            preds.append(_pred)
            _label = labels.cpu().data.numpy()
            test_y.append(_label)

        preds = [np.argmax(pred) for pred in preds]
        preds = np.asarray(preds).flatten()
        test_y = np.asarray(test_y).flatten()
        test_y = [test_dataset.id2str[_y] for _y in test_y]
        preds = [test_dataset.id2str[_y] for _y in preds]
        return test_y, preds


class TransformerEnsemble(nn.Module):
    def __init__(self, models_dir, folds=5, num_labels=4):
        super(TransformerEnsemble, self).__init__()
        self.models = nn.ModuleList([])
        self.last_layer_shape = None
        for fold in range(1, folds + 1):
            model_path = Path(f'{models_dir}{fold}')
            config = AutoConfig(model_path, num_labels=num_labels)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
            if not self.last_layer_shape:
                self.last_layer_shape = model.fc.shape()
            model.fc = nn.Identity()
            self.models.append(model)
        self.classifier = nn.Layer(self.last_layer_shape)

    def forward(self, x):
        xs = []
        for i in range(len[self.models]):
            xi = self.models[i](x.clone())
            xs.append(xi.view(xi.size(0), -1))

        x = torch.cat((i for i in xs))
        x = self.classifier(nn.functional.relu(x))
        return x


class TrainerExperiment:
    @staticmethod
    def run():
        BERT_MODEL = 'bert-base-uncased'
        KFOLD = 5
        config = {
            'learning_rate': 2e-5,
            'num_train_epochs': 3
        }

        reports = []
        for fold in range(1, KFOLD + 1):
            train_dir = f'data/NELA/train_{fold}.tsv'
            test_dir = f'data/NELA/test_{fold}.tsv'
            output_dir = Path(f'results/nela/{fold}')
            model = Transformer(output_dir, BERT_MODEL)
            if not output_dir.exists():
                logger.info(f'Training {fold}')
                model.fit(config, train_dir)
            test_y, preds = model.eval(test_dir, output_dir)
            report = classification_report(y_true=test_y, y_pred=preds, digits=4, output_dict=True)

            data = {}
            for label in set(test_y):
                data['{key}_f1'.format(key=label)] = [round(report[label]['f1-score'] * 100, 2)]
            data['macro_f1'] = [round(report['macro avg']['f1-score'] * 100, 2)]

            logger.info(classification_report(y_true=test_y, y_pred=preds, digits=4))
            reports.append(pd.DataFrame.from_dict(data))

    @staticmethod
    def predict(test_path):
        BERT_MODEL = 'bert-base-uncased'
        model = TransformerEnsemble('results/nela')
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        test_dataset = NELADataset(test_path, tokenizer)
        test_iter = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)
        device = torch.device('cuda')
        model.to(device)
        model.eval()
        test_y = []
        preds = []

        for batch in tqdm(test_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # forward pass
            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     labels=labels)

            # record preds, trues
            _pred = logits.cpu().data.numpy()
            preds.append(_pred)
            _label = labels.cpu().data.numpy()
            test_y.append(_label)

        preds = [np.argmax(pred) for pred in preds]
        preds = np.asarray(preds).flatten()
        test_y = np.asarray(test_y).flatten()
        test_y = [test_dataset.id2str[_y] for _y in test_y]
        preds = [test_dataset.id2str[_y] for _y in preds]
        report = classification_report(y_true=test_y, y_pred=preds, digits=4, output_dict=True)
        logger.info(report)


if __name__ == '__main__':
    TrainerExperiment.run()
