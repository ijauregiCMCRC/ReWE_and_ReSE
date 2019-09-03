sgm_file=open('IWSLT_2016/french_data/out_of_domain/newstest2014-fren-src.fr.sgm')

text_file=open('IWSLT_2016/french_data/out_of_domain/newstest2014.fr','w')

for line in sgm_file:
    if "<seg id=" in line:
        new_line=line.replace('</seg>','')
        new_line = new_line.replace('\n', '')
        new_line=new_line.split('">')
        if len(new_line)>2:
            sentence='">'.join(new_line[1:])
        else:
            sentence=new_line[1]

        text_file.write(sentence+'\n')