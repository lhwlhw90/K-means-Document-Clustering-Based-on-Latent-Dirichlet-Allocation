import pandas as pd

def print_top_words(model, feature_names, n_top_words, path):
    print('for reduce dimension')
    out = open(path, 'w')
    # model.components_ = lsa.inverse_transform(model.components_)
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        out.write(message+'\n')
    out.close()
    print()


def print_cluster(model, feature, n, path):
    print('for cluster')
    # model.cluster_centers_ = lsa.inverse_transform(model.cluster_centers_)
    out = open(path, 'w')
    for topic_idx, topic in enumerate(model.cluster_centers_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature[i]
                             for i in topic.argsort()[:-n - 1:-1]])
        print(message)
        out.write(message+'\n')
    out.close()
    print()


def output_result(corpus, result, path):
    df = pd.DataFrame({'text': corpus, 'label': result})
    df[['text', 'label']].to_csv(path, index=False)

