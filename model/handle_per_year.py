from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import time
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



sentences = []
stopwords = set()
n_topic = 10

with open('/home/lhw/PycharmProjects/nlp_pro/Chinese/stopwords.dat') as f:
    for line in f:
        stopwords.add(line.strip())

start_year = 2005
Abstract_file = open('/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract')

for idx, lines in enumerate(Abstract_file):
    if idx % 3 == 0:
        year = []
        continue
    if len(lines) == 0:
        continue
    lines = lines.strip().split("@")
    if idx % 3 == 1:
        for line in lines:
            year.append(' '.join(list(jieba.cut(line.strip()))))
        sentences.append(year)

for i in sentences:
    print(i)


for idx, i in enumerate(sentences):
    print("year:", start_year+idx)
    if len(i) > 1:
        vectorizer = TfidfVectorizer(max_features=3000,stop_words=stopwords)
        dataset = vectorizer.fit_transform(i)
        t = time.time()
        print(time.time() - t)
        print('Lda')
        t = time.time()
        lda = LatentDirichletAllocation(n_components=n_topic, verbose=1, learning_method='batch', max_iter=100)
        dataset_reduced = lda.fit_transform(dataset)
        print(time.time() - t)

        thershold = int((dataset.shape[0] // n_topic) + 0.00 * dataset.shape[0])

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
        print(np.argmax(dataset_km, -1))
        # print(dataset)
        # print(km.cluster_centers_)
        # print(lda.components_)

        # cv = cross_val_score(KMeans(), X=dataset, y=None, scoring=km.score,verbose=1, cv=10)
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







