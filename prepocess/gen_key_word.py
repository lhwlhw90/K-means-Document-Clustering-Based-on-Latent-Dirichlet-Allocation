import os

out_c = open('year_keyword_c', 'w')
out_e = open('year_keyword_e', 'w')
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
                if line[:3] == u'关键词':
                    key_word.extend(line[4:].replace(',', '，').split('，'))
                if line[:9] == u'KEY WORDS':
                    english_key_word.extend(line[10:].replace('，', ',').split(','))

    year_key_word[i] = key_word
    year_english_key_word[i] = english_key_word

for i in year_key_word.keys():
    out_c.write(i[0:-4]+'\n')
    for j in year_key_word[i]:
        out_c.write(j+' ')
    out_c.write("\n")

for i in year_english_key_word.keys():
    out_e.write(i[0:-4] + '\n')
    for j in year_english_key_word[i]:
        out_e.write(j + ' ')
    out_e.write('\n')

out_c.close()
out_e.close()

