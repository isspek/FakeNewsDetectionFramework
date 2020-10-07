import pandas as pd 
from pathlib import Path


DATA_DIR = Path('../data')

def read_constraint_splits():
    train_fpath = DATA_DIR / 'train.tsv'
    val_fpath = DATA_DIR / 'val.tsv'
    train = pd.read_csv(train_fpath, sep='\t')
    val = pd.read_csv(val_fpath,sep='\t')
    return {
        'train': train,
        'val': val
    }