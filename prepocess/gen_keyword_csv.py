# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys

out = open('keyword_theme.csv', 'w')
out.write('key_word,E_key_word,abstract,E_abstract'+'\n')
fl = os.listdir(r"/home/lhw/PycharmProjects/nlp_pro/data/dataset/txt")
print(fl)
for i in fl:
    with open(r"/home/lhw/PycharmProjects/nlp_pro/data/dataset/txt/" + i, encoding='utf8') as f:
        records = f.read().split('\n\n\n')
        for record in records:
            one_output = []
            for line in record.split('\n'):
                line = line.strip()
                if len(line) != 1:
                    # print(line + '\n')
                    if line[:3] == u"关键词":
                        one_output.append(line)
                    if line[:9] == u'KEY WORDS':
                        one_output.append(line)
                    if line[:3] == u'摘要':
                        one_output.append(line)
                    if line[:8] == u'ABSTRACT':
                        one_output.append(line)
            if one_output:
                print(one_output)
                out.write(' '.join(one_output)+'\n')







