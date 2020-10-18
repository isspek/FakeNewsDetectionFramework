import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import sqlite3

DATA_DIR = Path('data')


def read_constraint_splits():
    train_fpath = DATA_DIR / 'train.tsv'
    val_fpath = DATA_DIR / 'val.tsv'
    train = pd.read_csv(train_fpath, sep='\t')
    val = pd.read_csv(val_fpath, sep='\t')
    return {
        'train': train,
        'val': val
    }


def process_nela(nela_2018_path: str, nela_2019_path: str):
    '''
    NELA datasets have the following columns:
        - date
        - source
        - name (title)
        - content
    '''
    nela_2018_dir = Path(nela_2018_path)
    nela_2018_cnx = sqlite3.connect(nela_2018_dir / 'articles.db')
    nela_2018 = pd.read_sql_query("SELECT * FROM articles", nela_2018_cnx)
    print(nela_2018.columns)

    nela_2019_dir = Path(nela_2019_path)
    nela_2019_cnx = sqlite3.connect(nela_2019_dir / 'articles.db')
    nela_2019 = pd.read_sql_query("SELECT * FROM newsdata", nela_2019_cnx)
    print(nela_2019.columns)


    # labels_path = nela_2019_dir / 'labels.csv'
    # labels = pd.read_csv(labels_path)
    #
    # # use only Open Sources labels
    # labels = labels[['Unnamed: 0', 'Open Sources, reliable', 'Open Sources, fake',
    #                  'Open Sources, unreliable', 'Open Sources, bias',
    #                  'Open Sources, conspiracy', 'Open Sources, hate',
    #                  'Open Sources, junksci', 'Open Sources, rumor', 'Open Sources, blog',
    #                  'Open Sources, clickbait', 'Open Sources, political',
    #                  'Open Sources, satire', 'Open Sources, state']]
    #
    # labels_dict = {}
    # for _, row in labels.iterrows():
    #     source = row['Unnamed: 0'].lower()
    #     labels_dict[source] = []
    #     if row['Open Sources, reliable'] >= 2.0:
    #         labels_dict[source].append('reliable')
    #     if row['Open Sources, fake'] >= 2.0:
    #         labels_dict[source].append('fake')
    #     if row['Open Sources, bias'] >= 2.0:
    #         labels_dict[source].append('bias')
    #     if row['Open Sources, conspiracy'] >= 2.0:
    #         labels_dict[source].append('conspiracy')
    #     if row['Open Sources, conspiracy'] >= 2.0:
    #         labels_dict[source].append('conspiracy')


if __name__ == '__main__':
    parser = ArgumentParser()

    # NELA settings
    parser.add_argument('--nela_2018', type=str)
    parser.add_argument('--nela_2019', type=str)

    args = parser.parse_args()

    if args.nela_2018 and args.nela_2019:
        process_nela(args.nela_2018, args.nela_2019)
