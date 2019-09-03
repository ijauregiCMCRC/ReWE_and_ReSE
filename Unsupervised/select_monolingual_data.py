import random


file_en=open('data/mono/all.en')

file_fr=open('data/mono/all.fr')


all_en =[]
for line in file_en:
    all_en.append(line)

all_fr =[]
for line in file_fr:
    all_fr.append(line)

print (len(all_en))
print (len(all_fr))

ids = random.sample(range(0,len(all_en)),5000000)

new_file_en = open('data/mono_5.000.000/all.en','w')
new_file_fr = open('data/mono_5.000.000/all.fr','w')
for id in ids:
    new_file_en.write(all_en[id])
    new_file_fr.write(all_fr[id])

