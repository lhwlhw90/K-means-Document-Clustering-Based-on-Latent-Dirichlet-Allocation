from gensim.summarization import keywords
import jieba
import os


def cut(text):
    return ' '.join(jieba.cut(text))


out = open('rank_kw', 'w')
f = open('/home/lhw/PycharmProjects/nlp_pro/prepocess/abstract')
lines = f.readlines()
group = []
groups = []
for i, s in enumerate(lines):
    group.append(s)
    if (i+1) % 3 == 0:
        print(group)
        groups.append(group)
        group = []

for i in groups:
    if i[2] != '\n' and i[1] != '\n':
        c = keywords(cut(i[1]), lemmatize=True)
        e = keywords(cut(i[2]), lemmatize=True)
        c = ' '.join(c.split('\n'))
        e = ' '.join(e.split('\n'))
        out.write(i[0])
        out.write(c + '\n')
        out.write(e + '\n')
    else:
        out.write(i[0])
        out.write('\n')
        out.write('\n')
out.close()










