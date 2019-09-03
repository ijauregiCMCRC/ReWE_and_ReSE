import gensim, logging
import os
import nltk
import codecs

class MySentences(object):
    def __init__(self,dirname):
        self.dirname=dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in codecs.open(os.path.join(self.dirname,fname),encoding='utf-8'):
                #Maybe we need to use the nltk tokenizer here
                line2=nltk.word_tokenize(line)
                yield line2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#1. Store the sentences
sentences=MySentences('WMT17_news_data/word2vec_embeddings/text_for_embe/europarl/english')

#2. Learn the Model
model=gensim.models.Word2Vec(sentences,size=500, min_count=1)

#3. Save the model
model.wv.save_word2vec_format('WMT17_news_data/word2vec_embeddings/pre_train_embe/europarl/europarl.500d.en.txt',binary=False)
