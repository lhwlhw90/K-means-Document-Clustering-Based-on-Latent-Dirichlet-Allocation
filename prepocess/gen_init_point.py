import numpy as np


def gen_init_point(docTopic, docWord, a):

    thershold = int((docTopic.shape[0] // docTopic.shape[1]) + a * docTopic.shape[0])

    print(thershold)

    topic_mean = docTopic.mean(axis=0)
    print(topic_mean)
    # print(topic_mean)

    support_doc_n = []
    support_doc_index = []

    for x in range(docTopic.shape[1]):
        topic = docTopic[:, x]
        res_list = topic > topic_mean[x]
        res_index = np.where(res_list == True)
        support_doc_n.append(len(res_index[0]))
        support_doc_index.append(res_index[0])

    # print(support_doc_index)
    # print(support_doc_n)
    support_doc_n = np.array(support_doc_n)
    # print(support_doc_n)
    typical_topic = np.where(support_doc_n > thershold)[0]
    # print(typical_topic)
    # print(typical_topic)

    k_clustering_init = []
    for i in typical_topic:
        # print(i)
        # print(support_doc_index[i])
        # print(dataset[support_doc_index[i]])
        k_clustering_init.append(np.asarray(docWord[support_doc_index[i]].mean(axis=0)).reshape(-1))
    return k_clustering_init
