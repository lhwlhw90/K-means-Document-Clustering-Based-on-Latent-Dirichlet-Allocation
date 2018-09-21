from model import Preprocess
from sklearn.decomposition import LatentDirichletAllocation, NMF
from model import pipeline


print('LDA')
print('first we look chinese all years')

n_topic = 20
data, feature, corpus = Preprocess.preprocess_chinese()
pipeline.pipeline(LatentDirichletAllocation,n_topic,data,0.1,corpus,feature,'all','lda_chinese')

print('then we look per year chinese')
n_topic = 10
data, year, feature, corpus = Preprocess.preprocess_per_year_chinese()
for idx, i in enumerate(data):
    pipeline.pipeline(LatentDirichletAllocation, n_topic, i, 0.00, corpus[idx], feature[idx], year[idx], 'lda_chinese')


print('we look at all english')
data, feature, corpus = Preprocess.preprocess_english()
pipeline.pipeline(LatentDirichletAllocation,n_topic,data,0.1,corpus,feature,'all','lda_english')

print('per year english')

data, year, feature, corpus = Preprocess.preprocess_per_year_english()

for i in corpus:
    print(i)
print(year)
for idx, i in enumerate(data):
    pipeline.pipeline(LatentDirichletAllocation,n_topic,i,0.0,corpus[idx],feature[idx], year[idx],'lda_english')






print('NMF')
print('first we look chinese all years')

n_topic = 20
data, feature, corpus = Preprocess.preprocess_chinese()
pipeline.pipeline(NMF, n_topic, data, 0.1, corpus, feature, 'all', 'nmf_chinese')

print('then we look per year chinese')
n_topic = 10
data, year, feature, corpus = Preprocess.preprocess_per_year_chinese()
for idx, i in enumerate(data):
    pipeline.pipeline(NMF, n_topic, i, 0.02, corpus[idx], feature[idx], year[idx], 'nmf_chinese')

print('we look at all english')
data, feature, corpus = Preprocess.preprocess_english()
pipeline.pipeline(NMF, n_topic, data, 0.05, corpus, feature, 'all', 'nmf_english')

print('per year english')

data, year, feature, corpus = Preprocess.preprocess_per_year_english()
for idx, i in enumerate(data):
    pipeline.pipeline(NMF, n_topic, i, 0.1, corpus[idx], feature[idx], year[idx], 'nmf_english')








