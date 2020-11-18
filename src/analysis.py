from pathlib import Path
import pandas as pd
import csv
from sklearn.metrics import classification_report

RESULTS = 'results_style'

model = pd.read_csv(Path(RESULTS) / 'val_results.csv')
confusion_matrix = pd.crosstab(model['labels'], model['predictions'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print(classification_report(model['labels'], model['predictions'], digits=4))

labels = model['labels'].to_numpy()
preds = model['predictions'].to_numpy()

# data = pd.read_csv('data/val.tsv', quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
# tweets = data['tweet'].to_list()
# print(len(tweets))
# ids = data['id'].tolist()
# print(len(ids))
# answers = dict(zip(ids, preds))
# with open('data/answer.txt', 'w') as the_file:
#     the_file.write(f'id,label\n')
#     # for key, value in enumerate(preds):
#     # the_file.write(f'{key+1},{value}\n')
#     for key, value in answers.items():
#         the_file.write(f'{key},{value}\n')
