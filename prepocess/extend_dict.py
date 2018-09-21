out = open('common word', 'w')
with open('/home/lhw/PycharmProjects/nlp_pro/prepocess/rank_kw') as f:
    for idx, line in enumerate(f):
        if idx % 3 == 0:
            continue
        else:
            line = line.split(" ")
            for i in line:
                out.write(i + '\n')

out.close()