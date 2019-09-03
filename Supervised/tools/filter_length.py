read_file_en=open('./WMT17_news_data/Parallel_data/europarl-v7.de-en.tok.en','r')
read_file_de=open('./WMT17_news_data/Parallel_data/europarl-v7.de-en.tok.de','r')

write_file_en=open('./WMT17_news_data/Parallel_data/europarl-v7.de-en.tok.50l.en','w')
write_file_de=open('./WMT17_news_data/Parallel_data/europarl-v7.de-en.tok.50l.de','w')


list_en=[]
for line in read_file_en:
    list_en.append(line)

list_de=[]
for line in read_file_de:
    list_de.append(line)


for i in range(len(list_en)):
    if len(list_en[i].split()) <= 50 and len(list_de[i].split()) <= 50:
        write_file_en.write(list_en[i])
        write_file_de.write(list_de[i])
