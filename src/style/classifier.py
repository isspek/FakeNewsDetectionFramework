import torch
import pandas as pd
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


class NELADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.str2id = {'reliable': 0, 'mixed': 1, 'unreliable': 2, 'satire': 3}
        self.id2str = {val: key for key, val in self.str2id.items()}
        data = pd.read_csv(path, sep='\t')[:10000]
        self.size = len(data)
        data['title'] = data.title.map(lambda x: self.clean_helper(x))
        data['content'] = data.content.map(lambda x: self.clean_helper(x))
        data['label'] = data.label.map(lambda x: self.str2id[x])
        title, content = data['title'].tolist(), data['content'].tolist()
        self.encodings = tokenizer(title, content, padding=True, truncation='longest_first', verbose=False,
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
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        temp_dir = Path(self.output_dir) / 'temp'
        training_args = TrainingArguments(
            output_dir=temp_dir,  # output directory
            overwrite_output_dir=True,
            do_train=True,
            learning_rate=config['learning_rate'],
            num_train_epochs=config['num_train_epochs'],  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=1,  # batch size for evaluation
            warmup_steps=1000,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=1000,
            load_best_model_at_end=True,
            save_steps=1000,
            save_total_limit=1
        )

        set_seed(training_args.seed)

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
        )

        trainer.train()

        shutil.rmtree(temp_dir)
        trainer.save_model(self.output_dir)

    def eval(self, test_dir, model_path):
        test_dataset = NELADataset(test_dir, self.tokenizer)

        device = torch.device('cuda')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        test_iter = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)

        model.to(device)
        model.eval()
        test_y = []
        preds = []

        for batch in tqdm(test_iter):
            input_ids, att_masks, labels = batch

            input_ids = input_ids.to(device)
            att_masks = att_masks.to(device)
            labels = labels.to(device)

            # forward pass
            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

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


class KFoldExperiment:
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
        KFOLD = 5
        for fold in range(KFOLD):
            output_dir = Path(f'results/nela/{fold}')
            model = Transformer(output_dir, BERT_MODEL)
            test, y_pred = model.eval(test_path, output_dir)
            assert NotImplemented


if __name__ == '__main__':
    KFoldExperiment.run()
