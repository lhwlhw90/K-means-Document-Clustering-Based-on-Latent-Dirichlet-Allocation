from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import jieba
import string
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import NMF
import numpy as np
jieba.load_userdict("/home/lhw/PycharmProjects/nlp_pro/prepocess/dict")
# jieba.load_userdict("/home/lhw/PycharmProjects/nlp_pro/Chinese/dict.dat")
t = time.time()
print('load_data')
keywords = set()
stopwords = set()
stopwords.add('结果表明')
stopwords.add('实际上')
stopwords.add("方法")
stopwords.add('nt')
stopwords.add("：")
stopwords.add('th')
stopwords.add('er')
stopwords.add('ed')
stopwords.add('50')
stopwords.add('40')
sentences = []
Abstract_file = open('/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract')
commonwords = set()
jieba.add_word("desulphurization")


with open('/home/lhw/PycharmProjects/nlp_pro/Chinese/stopwords.dat') as f:
    for line in f:
        stopwords.add(line.strip())

with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/dict_c') as f:
    for line in f:
        keywords.add(line.strip())

# with open('/home/lhw/PycharmProjects/nlp_pro/Chinese/dict.dat') as f:
#     for line in f:
#         commonwords.add(line.strip())
#
# keywords = keywords.union(commonwords)


for idx, lines in enumerate(Abstract_file):
    if idx % 3 == 0:
        continue
    if len(lines) == 0:
        continue
    lines = lines.strip().split("@")
    if idx % 3 == 1:
        for line in lines:
            sentences.append(' '.join(list(jieba.cut(line.strip()))))

# print(dataset.data)
print('key words num:', len(keywords))
n_topic = 10
print('vectorizer')
t = time.time()
vectorizer = TfidfVectorizer(vocabulary=keywords, max_features=3000, stop_words=stopwords)
dataset = vectorizer.fit_transform(sentences)
print('shape:')
print(dataset.shape)
# lsa = NMF(n_components=500, verbose=1, max_iter=10)
# dataset = lsa.fit_transform(dataset)
# explained_variance = lsa.reconstruction_err_
# print("Explained variance of the SVD step: {}%".format(explained_variance))
# print(dataset.shape)
print(time.time() - t)
print('Lda')
t = time.time()
lda = LatentDirichletAllocation(n_components=n_topic, verbose=1, max_iter=100)
dataset_reduced = lda.fit_transform(dataset)
print(time.time() - t)
# print(dataset)

thershold = int((dataset.shape[0] // n_topic) + 0.10 * dataset.shape[0])

print(thershold)

topic_mean = dataset_reduced.mean(axis=0)
print(topic_mean)
# print(topic_mean)

support_doc_n = []
support_doc_index = []

for x in range(n_topic):
    topic = dataset_reduced[:, x]
    res_list = topic > topic_mean[x]
    res_index = np.where(res_list == True)
    support_doc_n.append(len(res_index[0]))
    support_doc_index.append(res_index[0])


# print(support_doc_index)
# print(support_doc_n)
support_doc_n = np.array(support_doc_n)
print(support_doc_n)
typical_topic = np.where(support_doc_n > thershold)[0]
# print(typical_topic)
print(typical_topic)

k_clustering_init = []
for i in typical_topic:
    # print(i)
    # print(support_doc_index[i])
    # print(dataset[support_doc_index[i]])
    k_clustering_init.append(np.asarray(dataset[support_doc_index[i]].mean(axis=0)).reshape(-1))

# print(k_clustering_init)
# print(dataset.shape)
print('clusters:', len(k_clustering_init))
km = KMeans(n_clusters=len(k_clustering_init), init=np.array(k_clustering_init), n_init=1, verbose=1)
dataset_km = km.fit_transform(dataset)
result_clusters = np.argmax(dataset_km, -1)
result = pd.DataFrame({'sentences':sentences, 'cluster': result_clusters})
result[['sentences', 'cluster']].to_csv('result_cluster', index=False)

# print(dataset)
# print(km.cluster_centers_)
# print(lda.components_)

#cv = cross_val_score(KMeans(), X=dataset, y=None, scoring=km.score,verbose=1, cv=10)
# print(cv)
# print(cv.mean())

print('loss:', km.score(dataset))

def print_top_words(model, feature_names, n_top_words):
    # model.components_ = lsa.inverse_transform(model.components_)
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def print_cl(model, feature, n):
    # model.cluster_centers_ = lsa.inverse_transform(model.cluster_centers_)
    for topic_idx, topic in enumerate(model.cluster_centers_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature[i]
                             for i in topic.argsort()[:-n - 1:-1]])
        print(message)
    print()


print_top_words(lda, vectorizer.get_feature_names(), 10)
print_cl(km, vectorizer.get_feature_names(), 10)


lsa = TruncatedSVD(n_components=2)
data = lsa.fit_transform(dataset)
plt.scatter(data[:, 0], data[:, 1])
plt.show()
# plt.show()
# distance = cosine_distances(dataset)
# mds = MDS(n_components=2)
# data = mds.fit_transform(dataset.toarray())
#
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()
#
# pca = PCA(n_components=2)
# data = pca.fit_transform(dataset.toarray())
#
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()
#
# tse = TSNE(n_components=2)
# data = tse.fit_transform(dataset.toarray())
#
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()
#
# lle = LocallyLinearEmbedding(n_components=2, random_state=2018)
# data = lle.fit_transform(dataset.toarray())
#
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()
#
#
# km_new = KMeans(n_clusters=9)
# out_1 = km_new.fit_transform(data)
#
# print(out_1)
# out_1 = np.argmax(out_1, axis=1)
# print(out_1)
#
#
# print_cl(km_new, vectorizer.get_feature_names(), 10)

