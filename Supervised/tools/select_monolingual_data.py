import random


file_en=open('../WMT18/de_en/train_val/train.tok.en')

file_fr=open('../WMT18/de_en/train_val/train.tok.de')


all_en =[]
for line in file_en:
    all_en.append(line)

all_fr =[]
for line in file_fr:
    all_fr.append(line)

print (len(all_en))
print (len(all_fr))

ids = random.sample(range(0,len(all_en)),2000000)

new_file_en = open('../WMT18/de_en/train_val/2M/train.tok.en','w')
new_file_fr = open('../WMT18/de_en/train_val/2M/train.tok.de','w')
for id in ids:
    new_file_en.write(all_en[id])
    new_file_fr.write(all_fr[id])

