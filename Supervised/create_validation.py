file_2015de=open('WMT18/de_en/out_of_domain/test_2015.de')
file_2015en=open('WMT18/de_en/out_of_domain/test_2015.en')
file_2016de=open('WMT18/de_en/out_of_domain/test_2016.de')
file_2016en=open('WMT18/de_en/out_of_domain/test_2016.en')


write_val_de=open('WMT18/de_en/out_of_domain/test_2015_plus_2016.de','w')
write_val_en=open('WMT18/de_en/out_of_domain/test_2015_plus_2016.en','w')


list_2015_de=[]
for line in file_2015de:
    list_2015_de.append(line)
list_2015_en=[]
for line in file_2015en:
    list_2015_en.append(line)
list_2016_de=[]
for line in file_2016de:
    list_2016_de.append(line)
list_2016_en=[]
for line in file_2016en:
    list_2016_en.append(line)


for line in list_2015_de:
    write_val_de.write(line)
for line in list_2016_de:
    write_val_de.write(line)

for line in list_2015_en:
    write_val_en.write(line)
for line in list_2016_en:
    write_val_en.write(line)



