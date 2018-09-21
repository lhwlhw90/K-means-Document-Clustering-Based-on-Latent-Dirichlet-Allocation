from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

from sklearn.datasets import fetch_20newsgroups
import time
import numpy as np


print('load_data')
t = time.time()
dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
print(time.time() - t)

print(dataset.data)

n_topic = 20
print('vectorizer')
t = time.time()
vectorizer = TfidfVectorizer()
dataset = vectorizer.fit_transform(dataset.data)
print(dataset.shape)

km = KMeans(n_clusters=20, n_init=1)
dataset = km.fit_transform(dataset)
