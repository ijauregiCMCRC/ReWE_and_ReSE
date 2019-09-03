
src_file_data_big=open('IWSLT_2018/Basque_English/PaCo_EuEn_corpus/PaCo_corpus.en-eu.tok.eu')
tgt_file_data_big=open('IWSLT_2018/Basque_English/PaCo_EuEn_corpus/PaCo_corpus.en-eu.tok.en')
src_file_data_small=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/train.tok.eu')
tgt_file_data_small=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/train.tok.en')


list_src_file_data_small=[]
for line in src_file_data_small:
    #line_clean=line.replace('\n','')
    list_src_file_data_small.append(line)
list_tgt_file_data_small=[]
for line in tgt_file_data_small:
    #line_clean=line.replace('\n','')
    list_tgt_file_data_small.append(line)
list_src_file_data_big=[]
for line in src_file_data_big:
    #line_clean=line.replace('\n','')
    list_src_file_data_big.append(line)
list_tgt_file_data_big=[]
for line in tgt_file_data_big:
    #line_clean=line.replace('\n','')
    list_tgt_file_data_big.append(line)

print (len(list_src_file_data_small))
print (len(list_tgt_file_data_small))
print (len(list_src_file_data_big))
print (len(list_tgt_file_data_big))


new_src = list_src_file_data_big + list_src_file_data_small*10
new_tgt = list_tgt_file_data_big + list_tgt_file_data_small*10

print (len(new_src))
print (len(new_tgt))

new_src_file=open('IWSLT_2018/Basque_English/PaCo_EuEn_corpus/PaCo_plus_TED.tok.eu','w')
new_tgt_file=open('IWSLT_2018/Basque_English/PaCo_EuEn_corpus/PaCo_plus_TED.tok.en','w')

for sent in new_src:
    new_src_file.write(sent)

for sent in new_tgt:
    new_tgt_file.write(sent)

