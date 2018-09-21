from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import numpy as np
import matplotlib as plt
import pandas as pd
import time
import numpy as np
jieba.enable_parallel(4)
jieba.load_userdict("/home/lhw/PycharmProjects/nlp_pro/prepocess/dict")
# jieba.load_userdict("/home/lhw/PycharmProjects/nlp_pro/Chinese/dict.dat")
t = time.time()
print('load_data')
keywords = set()
stopwords = set()
stopwords.add('结果表明')
stopwords.add('实际上')
stopwords.add('50')
stopwords.add("方法")
stopwords.add('nt')
stopwords.add("：")
stopwords.add('th')
stopwords.add('er')
stopwords.add('ed')
sentences = []
Abstract_file = open('/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract')
commonwords = set()
jieba.add_word("desulphurization")




with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/dict_e') as f:
    for line in f:
        keywords.add(line.strip())

for idx, lines in enumerate(Abstract_file):
    if idx % 3 == 0:
        continue
    if len(lines) == 0:
        continue
    lines = lines.strip().split("@")
    if idx % 3 == 2:
        for line in lines:
            sentences.append(line.strip())

print(time.time() - t)

# print(dataset.data)
print('key words num:', len(keywords))
n_topic = 30
print('vectorizer')
t = time.time()
vectorizer = TfidfVectorizer(stop_words='english', vocabulary=keywords, max_features=3000)
dataset = vectorizer.fit_transform(sentences)
print('shape:')
print(dataset.shape)

