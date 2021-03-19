from argparse import ArgumentParser
from src.data_reader import DATA_READER
import numpy as np
import gensim
import pandas as pd
from matplotlib.ticker import FuncFormatter
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plot
import seaborn as sns

stopwords_en = stopwords.words('english')


# # Plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)
#
# # Topic Distribution by Dominant Topics
# ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
# ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
# tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
# ax1.xaxis.set_major_formatter(tick_formatter)
# ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
# ax1.set_ylabel('Number of Documents')
# ax1.set_ylim(0, 1000)
#
# # Topic Distribution by Topic Weights
# ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
# ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
# ax2.xaxis.set_major_formatter(tick_formatter)
# ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))
#
# plt.show()

def preprocess(text, stopwords_en=stopwords_en, lemmatizer=WordNetLemmatizer()):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub("\d+", " ", text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # remove duplicated words in row
    splitted_text = text.split()
    splitted_text = [lemmatizer.lemmatize(text, 'v') for text in splitted_text]
    splitted_text = [lemmatizer.lemmatize(text) for text in splitted_text]
    condition = lambda content: content not in stopwords_en
    splitted_text = list(filter(condition, splitted_text))
    return splitted_text


#
# dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)
#
# # Distribution of Dominant Topics in Each Document
# df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
# dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
# df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
#
# # Total Topic Distribution by actual weight
# topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
# df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
#
# # Top 3 Keywords for each Topic
# topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False)
#                                  for j, (topic, wt) in enumerate(topics) if j < 3]
#
# df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
# df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
# df_top3words.reset_index(level=0,inplace=True)


# # Sentence Coloring of N Sentences
# def topics_per_document(model, corpus, start=0, end=1):
#     corpus_sel = corpus[start:end]
#     dominant_topics = []
#     topic_percentages = []
#     for i, corp in enumerate(corpus_sel):
#         topic_percs, wordid_topics, wordid_phivalues = model[corp]
#         dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
#         dominant_topics.append((i, dominant_topic))
#         topic_percentages.append(topic_percs)
#     return(dominant_topics, topic_percentages)

def find_topics(args):
    num_topics = 20
    data = args.data
    topic_model_path = args.topic_model_path
    topic_model = gensim.models.LdaModel.load(topic_model_path)
    top_k = args.top_k

    vocabulary = topic_model.id2word


    # topics and documents that are appeared
    samples = DATA_READER[data](args.data_path, 'fake')
    samples = [preprocess(text) for text in samples]
    samples_bow = [vocabulary.doc2bow(doc) for doc in samples]
    topics_ids = np.zeros((num_topics))
    topics_percentages_fake = []
    for i, doc in enumerate(samples_bow):
        topics_percentage = np.zeros((num_topics))
        topics = topic_model[doc]

        for topic_id, topic_distribution in topics:
            topics_ids[topic_id] += 1
            topics_percentage[topic_id] = topic_distribution
        topics_percentages_fake.append(topics_percentage)

    print(f'{top_k} appeared topics in fake\n')
    print(np.argpartition(topics_ids, -top_k)[-top_k:])

    print("Normalized topics / documents ratio")
    print(topics_ids / len(samples))

    # topics and documents that are appeared
    samples = DATA_READER[data](args.data_path, 'true') # change to 'real' for constraint dataset
    samples = [preprocess(text) for text in samples]
    samples_bow = [vocabulary.doc2bow(doc) for doc in samples]
    topics_ids = np.zeros((num_topics))
    topics_percentages_true = []
    for i, doc in enumerate(samples_bow):
        topics_percentage = np.zeros((num_topics))
        topics = topic_model[doc]

        for topic_id, topic_distribution in topics:
            topics_ids[topic_id] += 1
            topics_percentage[topic_id] = topic_distribution
        topics_percentages_true.append(topics_percentage)

    assert len(topics_percentages_true) == len(samples_bow)

    print(f'{top_k} appeared topics in true\n')
    print(np.argpartition(topics_ids, -top_k)[-top_k:])

    print("Normalized topics / documents ratio")
    print(topics_ids / len(samples))

    topics_percentages_fake = np.mean(np.asarray(topics_percentages_fake), axis=0)
    topics_percentages_true = np.mean(np.asarray(topics_percentages_true), axis=0)

    print(topics_percentages_fake)
    print(topics_percentages_true)

    index = [f'T{i}' for i in range(20)]

    df = pd.DataFrame({'True': topics_percentages_true,

                       'Fake': topics_percentages_fake}, index=index)

    ax = df.plot.bar()
    ax.figure.savefig('topic_dist_coaid_article.png')


def topic_embed(text, topic_model, vocabulary, num_topics=20):
    text = preprocess(text)
    text_bow = vocabulary.doc2bow(text)
    topics = topic_model[text_bow]
    topic_vector = np.zeros((num_topics))
    for topic_id, weight in topics:
        topic_vector[topic_id] = weight
    return topic_vector


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--data_path')
    parser.add_argument('--topic_model_path')
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--target', type=str)
    args = parser.parse_args()

    find_topics(args)
