import os
out = open('abstract', 'w')
fl = os.listdir(r"/home/lhw/PycharmProjects/nlp_pro/data/dataset/txt")
print(fl)
year_key_word = {}
year_english_key_word = {}
for i in fl:
    with open(r"/home/lhw/PycharmProjects/nlp_pro/data/dataset/txt/" + i, encoding='utf8') as f:
        key_word = []
        english_key_word = []
        records = f.read().split('\n\n\n')
        for record in records:
            for line in record.split('\n'):
                if line[:2] == u'摘要':
                    key_word.append(line[3:])
                    print(line[3:])
                    print(key_word)
                if line[:8] == u'ABSTRACT':
                    english_key_word.append(line[9:])
                    print(line[9:])
                    print(english_key_word)

    year_key_word[i] = key_word
    year_english_key_word[i] = english_key_word

for i in year_key_word.keys():
    out.write(i[0:-4]+'\n')
    for j in year_key_word[i]:
        out.write(j+'@')
    out.write('\n')
    for j in year_english_key_word[i]:
        out.write(j + '@')
    out.write('\n')

out.close()

