from sklearn.cluster import KMeans
from model import gen_init_point
from model import helper
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score

def pipeline(model, n_topic, data, a, corpus, feature, year, name):
    lda = model(n_components=n_topic)
    data_lda = lda.fit_transform(data)
    n_init_clusters = gen_init_point.gen_init_point(data_lda, data, a)
    # print(n_init_clusters)
    km = KMeans(n_clusters=len(n_init_clusters), init=n_init_clusters)
    km.fit_transform(data)
    print('score')
    print(calinski_harabaz_score(data.toarray(), km.labels_))
    print(silhouette_score(data.toarray(), km.labels_))
    helper.print_top_words(lda, feature, 10, name+'_topic_word'.format(year))
    helper.print_cluster(km, feature, 10, name+'_cluster_meaning_{}'.format(year))
    helper.output_result(corpus, km.labels_, name+'_cluster_result_{}'.format(year))
