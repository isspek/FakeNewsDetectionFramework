from pathlib import Path
import pandas as pd
import csv
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

seeds = ['36','42','0']
# seeds = ['42']
# model = 'results_history_style'
model = 'results_history_style'
#model = 'results_style'
# model = 'results_style'

RESULTS = Path(model)

aggregated_preds = []
for idx, seed in enumerate(seeds):
    _data = pd.read_csv(str(seed) / RESULTS / 'test_results.csv')
    _preds = _data['predictions'].to_numpy()
    aggregated_preds.append(_preds)
aggregated_preds = np.amax(np.asarray(aggregated_preds), axis=0)
print(len(aggregated_preds))

data = pd.read_csv('data/groundtruth.tsv', quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
history = pd.read_csv('data/semantic_test_results.tsv', sep='\t')
groups = history.groupby(['tweet_id'])
processed_links = json.load(open('data/test_links_processed.json', 'r'))['links']
links = []
for idx, link in enumerate(processed_links):
    links.append(len(link))
# links = np.asarray([len(i) for i in links.values()])
links = np.asarray(links)
indices_having_links = np.where(links > 0)[0]

count_fake_links = 0
count_fake_no_links = 0
count_true_links = 0
count_true_no_links = 0
misclassified_ids = []
for idx, row in data.iterrows():
    if row['label'] != aggregated_preds[idx]:
        if 'http' in row['tweet']:
            if row['label'] == 'fake':
                count_fake_links += 1
                # print(idx)
                # print('=======')
                # print(row['label'])
                # print(row['tweet'])
            else:
                count_true_links += 1
                print(idx)
                print('=======')
                print(row['label'])
                print(row['tweet'])
        else:
            if row['label'] == 'fake':
                count_fake_no_links += 1
            else:
                count_true_no_links += 1
                count_true_links += 1
                print(idx)
                print('=======')
                print(row['label'])
                print(row['tweet'])
    #

print("Wrongly misclassified fake links")
print(count_fake_links)
print(count_fake_links + count_true_links)
print(count_fake_links / (count_fake_links + count_true_links))
print("Wrongly misclassified true links")
print(count_true_links)
print(count_fake_links + count_true_links)
print(count_true_links / (count_fake_links + count_true_links))
print("Wrongly misclassified fake no links")
print(count_fake_no_links)
print(count_fake_no_links + count_true_no_links)
print(count_fake_no_links / (count_fake_no_links + count_true_no_links))
print("Wrongly misclassified true no links")
print(count_true_no_links)
print(count_fake_no_links + count_true_no_links)
print(count_true_no_links / (count_fake_no_links + count_true_no_links))
prior_knowledge = pd.read_csv('data/semantic_test_results.tsv', sep='\t')
prior_knowledge = np.asarray([len(item) for item in prior_knowledge.iterrows()])
indices_having_prior_knowledge = np.where(prior_knowledge > 0)[0]
# data = data.tweet.tolist()
labels = data['label'].tolist()

print('Normal classification')
print(classification_report(labels, aggregated_preds, digits=4))
print(confusion_matrix(y_true=labels,y_pred=aggregated_preds))
print("Accuracy : ", accuracy_score(aggregated_preds, labels))
print("Precison : ", precision_score(aggregated_preds, labels, average='weighted'))
print("Recall : ", recall_score(aggregated_preds, labels, average='weighted'))
print("F1 : ", f1_score(aggregated_preds, labels, average='weighted'))

# classification among with links
print('classification among with links')
labels_links = [labels[i] for i in indices_having_links]
aggregated_preds_links = [aggregated_preds[i] for i in indices_having_links]
print(classification_report(labels_links, aggregated_preds_links, digits=4))
print("F1 : ", f1_score(aggregated_preds_links, labels_links, average='weighted'))

# # classification among with prior knowledge
# print('classification among with prior knowledge')
# labels_prior = [labels[i] for i in indices_having_prior_knowledge]
# aggregated_preds_prior = [aggregated_preds[i] for i in indices_having_prior_knowledge]
# print(classification_report(labels_prior, aggregated_preds_prior, digits=4))
# print("F1 : ", f1_score(aggregated_preds_prior, labels_prior, average='weighted'))

indices_without_links = np.where(links == 0)[0]
labels_no_links = [labels[i] for i in indices_without_links]
print('classification among with no link')
aggregated_no_links = [aggregated_preds[i] for i in indices_without_links]
print(classification_report(labels_no_links, aggregated_no_links, digits=4))
print("F1 : ", f1_score(labels_no_links, aggregated_no_links, average='weighted'))

# false_idx = [10, 13, 61, 145, 194, 202, 305, 319, 438, 453, 482, 559, 609, 728, 769, 813, 827, 838, 893, 921, 927, 965,
#              1017, 1061, 1181, 1185, 1217, 1218, 1312, 1356, 1435, 1440, 1521, 1553, 1566, 1577, 1586, 1591, 1629, 1677,
#              1685, 1765, 1769, 1772, 1790, 1908, 1920, 1930, 1991, 2107]
# # # print(len(false_idx))
# # # true = 0
# for i in false_idx:
#     print(data[i])
#     if aggregated_preds[i] == labels[i]:
#         true += 1
# print(true / len(false_idx))

# data = pd.read_csv('data/test.tsv', quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
# tweets = data['tweet'].to_list()
# print(len(tweets))
# ids = data['id'].tolist()
# print(len(ids))
# # answers = dict(zip(ids, preds))
#
# answers = []
# for idx, pred in enumerate(aggregated_preds):
#     answers.append({'id': ids[idx], 'label': pred})
#
# pd.DataFrame(answers).to_csv('ibaris_3.csv', columns=['id', 'label'], sep=',', index=False)
#
# answers = dict(zip(ids, aggregated_preds))
# with open('answer.txt', 'w') as the_file:
#     the_file.write(f'id,label\n')
#     # for key, value in enumerate(preds):
#     # the_file.write(f'{key+1},{value}\n')
#     for key, value in answers.items():
#         the_file.write(f'{key},{value}\n')
