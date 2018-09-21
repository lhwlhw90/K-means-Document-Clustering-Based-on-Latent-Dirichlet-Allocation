from model import Preprocess
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

print('only we look whole distribution')

from sklearn.manifold import LocallyLinearEmbedding
from model import gen_init_point


data, feature, corpus = Preprocess.preprocess_chinese()


n_topic = 20
lle = LocallyLinearEmbedding(n_components=2)
data_lle = lle.fit_transform(data)
plt.show(data[:,0],data[:,1])
plt.show()



km = KMeans(n_cluster=)

