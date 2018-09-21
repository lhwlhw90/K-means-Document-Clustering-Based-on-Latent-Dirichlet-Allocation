all_words = []
stopwords = set()
with open('/home/lhw/PycharmProjects/nlp_pro/Chinese/stopwords.dat') as f:
    for line in f:
        stopwords.add(line.strip())

with open("year_keyword_e") as f:
    for line in f:
        line = line.split(' ')
        line = list(map(str.strip, line))
        all_words.extend(line)

all_words = list(set(all_words) -  stopwords)
with open("dict_e", 'w') as f:
    for i in all_words:
        if i != ' ':
            f.write(i+'\n')





