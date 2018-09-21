from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
jieba.enable_parallel(4)
jieba.load_userdict("/home/lhw/PycharmProjects/nlp_pro/prepocess/dict")


def preprocess_english():
    keywords = set()
    sentences = []

    with open("/home/lhw/PycharmProjects/nlp_pro/prepocess/common word") as f:
        for line in f:
            keywords.add(line.strip())
    with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/dict_e') as f:
        for line in f:
            keywords.add(line.strip())

    for idx, lines in enumerate(open('/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract')):
        if idx % 3 == 0:
            continue
        if len(lines) == 0:
            continue
        lines = lines.strip().split("@")
        if idx % 3 == 2:
            for line in lines:
                sentences.append(line.strip())
    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=keywords, max_features=3000)
    dataset = vectorizer.fit_transform(sentences)

    return dataset, vectorizer.get_feature_names(), sentences


def preprocess_chinese():

    stopwords = set()
    keywords = set()
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

    with open("/home/lhw/PycharmProjects/nlp_pro/prepocess/common word") as f:
        for line in f:
            keywords.add(line.strip())

    with open('/home/lhw/PycharmProjects/nlp_pro/Chinese/stopwords.dat') as f:
        for line in f:
            stopwords.add(line.strip())

    with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/dict_c') as f:
        for line in f:
            keywords.add(line.strip())

    for idx, lines in enumerate(open("/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract")):
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
    print('vectorizer')
    vectorizer = TfidfVectorizer(vocabulary=keywords, max_features=3000, stop_words=stopwords)
    dataset = vectorizer.fit_transform(sentences)
    return dataset, vectorizer.get_feature_names(), [i.replace(' ', '') for i in sentences]


def preprocess_per_year_chinese():
    sentences = []
    stopwords = set()
    year = []
    start_year = 2005
    year_index = []
    res = []
    keywords = set()
    stopwords.add('结果表明')
    stopwords.add('实际上')
    stopwords.add('50')
    stopwords.add("方法")
    stopwords.add('nt')
    stopwords.add("：")
    stopwords.add('th')
    stopwords.add('er')
    stopwords.add('ed')
    feature = []
    corpus = []

    with open('/home/lhw/PycharmProjects/nlp_pro/Chinese/stopwords.dat') as f:
        for line in f:
            stopwords.add(line.strip())
    with open("/home/lhw/PycharmProjects/nlp_pro/prepocess/common word") as f:
        for line in f:
            keywords.add(line.strip())
    with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/dict_c') as f:
        for line in f:
            keywords.add(line.strip())

    for idx, lines in enumerate(open("/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract")):
        if idx % 3 == 0:
            year = []
            continue
        if len(lines) == 0:
            continue
        lines = lines.strip().split("@")
        if idx % 3 == 1:
            for line in lines:
                if len(line) > 10:
                    year.append(' '.join(list(jieba.cut(line.strip()))))
            sentences.append(year)

    for idx, i in enumerate(sentences):
        # print("year:", start_year + idx)
        # print(i)
        if i:
            vectorizer = TfidfVectorizer(vocabulary=keywords, max_features=3000, stop_words=stopwords)
            dataset = vectorizer.fit_transform(i)
            year_index.append(start_year + idx)
            res.append(dataset)
            feature.append(vectorizer.get_feature_names())
            corpus.append(i)

    return res, year_index, feature, [[j.replace(" ", '') for j in i] for i in corpus]


def preprocess_per_year_english():
    sentences = []
    year = []
    start_year = 2005
    year_index = []
    res = []
    feature = []
    keywords = set()
    corpus = []

    with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/dict_e') as f:
        for line in f:
            keywords.add(line.strip())

    for idx, lines in enumerate(open("/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract")):
        if idx % 3 == 0:
            year = []
            continue
        if len(lines) == 0:
            continue
        if idx % 3 == 2:
            lines = lines.strip().split("@")
            for line in lines:
                if len(line) > 10:
                    year.append(line.strip())
            sentences.append(year)

    for idx, i in enumerate(sentences):
        # print("year:", start_year + idx)
        # print(i)
        if i:
            vectorizer = TfidfVectorizer(vocabulary=keywords, max_features=3000, stop_words='english')
            dataset = vectorizer.fit_transform(i)
            year_index.append(start_year + idx)
            res.append(dataset)
            feature.append(vectorizer.get_feature_names())
            corpus.append(i)

    return res, year_index, feature, corpus
