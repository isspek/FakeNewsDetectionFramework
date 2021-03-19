from pathlib import Path
import pandas as pd
import csv
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tqdm
import itertools

SEEDS = ['36']
#MODEL = 'results_style_coaid_article'
MODEL = 'results_style_ent_topic_pk_coaid_article'
RESULTS_FILE = 'test_results.csv'
RESULTS = Path(MODEL)
LABELS_FILE = 'data/coaid/article/36/test.tsv'
# PK_FILE = 'data/coaid/article/0/coaid_semantic_test_results.tsv'
# COS_FILE = 'data/coaid/article/0/0_results_pk_style_coaid_article_cos_sims.csv'

# LABELS_FILE = 'data/coaid/article/36/test.tsv'

#labels = pd.read_csv(LABELS_FILE, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')['label'].tolist()
labels = pd.read_csv(LABELS_FILE, sep='\t')['label'].tolist()
labels = np.asarray(['true' if label == 'real' else label for label in labels])
true_idx = np.where(labels == 'fake')
false_idx = np.where(labels != 'true')
preds = []

for idx, seed in enumerate(SEEDS):
    _data = pd.read_csv(str(seed) / RESULTS / RESULTS_FILE)
    _preds = _data['predictions'].to_numpy()
    preds.append(_preds)
    #
    # print(min(pk.groupby(['tweet_id']).count()))

    # for i, row in tqdm(_data.iterrows(), total=len(_data)):
    #     similar_false_claims = train_history_df[train_history_df['tweet_id'] == i]
    #     similar_false_claims = similar_false_claims.fillna('')
    #     similar_false_claims = similar_false_claims['title'].to_numpy() + similar_false_claims['content'].to_numpy()



preds = np.amax(np.asarray(preds), axis=0)

print('Normal classification')
print(classification_report(labels, preds, digits=4))
print(confusion_matrix(y_true=labels, y_pred=preds))
print("Accuracy : ", accuracy_score(preds, labels))
print("Precison : ", precision_score(preds, labels, average='weighted'))
print("Recall : ", recall_score(preds, labels, average='weighted'))
print("F1 macro: ", f1_score(preds, labels, average='macro'))
print("F1 weighted: ", f1_score(preds, labels, average='weighted'))


pk = pd.read_csv(PK_FILE, sep='\t')
cos_sims = pd.read_csv(COS_FILE, sep='\t', skiprows=[0], header=None, index_col=False).to_numpy().flatten()
true_cos_sims = np.asarray(cos_sims[true_idx])

investigated_ids = np.argwhere(true_cos_sims > 0.8).flatten()
true_idx = list(itertools.chain.from_iterable(true_idx))


for i in investigated_ids:
    idx = true_idx[i]
    query = pk[pk['tweet_id'] == idx]['query'].unique()
    print(f'QUERY {idx}')
    print(query)
    print('CONTENT')
    contents = pk[pk['tweet_id'] == idx]['content'].values[0]
    print(contents)
    print('PREDICTED')
    print(preds[0][idx])
    print('LABEL')
    print(labels[idx])

false_cos_sims = cos_sims[false_idx]
print(false_cos_sims)
print(f'True Max similarity {max(true_cos_sims)}')
print(f'True Min similarity {min(true_cos_sims)}')
print(f'True Mean similarity {np.mean(true_cos_sims)}')
print(f'True Variance {np.var(true_cos_sims)}')
print(f'True Std {np.std(true_cos_sims)}')
print(f'False Max similarity {max(false_cos_sims)}')
print(f'False Min similarity {min(false_cos_sims)}')
print(f'False Mean similarity {np.mean(false_cos_sims)}')
print(f'False Variance {np.var(false_cos_sims)}')
print(f'False Std {np.std(false_cos_sims)}')

# import matplotlib.pyplot as plt
#
# plt.hist(true_cos_sims, bins=20, range=(-1, 1), histtype='step', density=True, label='True')
# plt.hist(false_cos_sims, bins=20, range=(-1, 1), histtype='step',density=True, label='Fake')
# plt.legend(loc='upper left')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Sample Size (Normalized)')
# plt.savefig('constraint_proposed_model.png')


for idx, label in enumerate(labels):
    if label != preds[0][idx]:
        print(idx)

investigated_id = 750
print(preds[0][investigated_id])
print(labels[investigated_id])
query = pk[pk['tweet_id'] == investigated_id]['query'].unique()
print('QUERY')
print(query)
print('CONTENT')
contents = pk[pk['tweet_id'] == investigated_id]
print(contents)

