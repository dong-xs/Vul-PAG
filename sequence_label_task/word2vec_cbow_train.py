from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import pandas as pd

data=pd.read_csv('../generate_data/filtered_cve_item_14W.csv')
sent=data['description']

with open('word2vec_train_sentence.txt','w',encoding='utf-8') as f:
    for temp_sent in sent:
        f.write(temp_sent+'\n')
f.close()

model=Word2Vec(sentences='word2vec_train_sentence.txt',size=200,window=5,min_count=1)
model.save('word2vec.model')