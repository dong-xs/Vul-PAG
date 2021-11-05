#encoding:utf-8

from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import random
import time

import pandas as pd
#准备数据
data=pd.read_csv('../generate_data/filtered_cve_item_14W.csv')
sent=data['description']

with open('word2vec_train_sentence.txt','w',encoding='utf-8') as f:
    for temp_sent in sent:
        f.write(temp_sent+'\n')
f.close()

sentences=word2vec.LineSentence('word2vec_train_sentence.txt')

#设置参数
# num_features=100
# min_word_count=1
# num_workers=2
# window_size=5
# subsampling=1e-3
#
# model=Word2Vec(sentences,workers=num_workers,vector_size=num_features,min_count=min_word_count,window=window_size,sample=subsampling)
#
# model.wv.save_word2vec_format('security_domain_word2vec_model'+'.bin',binary=True)

# model.build_vocab(token_review)
#
# idx=list(range(len(token_review)))
# t0=time()
# for epoch in range(300):
#     print(epoch+1,'/300 epoch')
#     random.shuffle(idx)
#     perm_sentences=[token_review[i] for i in idx]
#     model.train(perm_sentences,total_examples=len(idx),epoch=1)
#
# elapsed=time()-t0
# print('time taken for word2vec training:',elapsed/60,' mins')

# model.wv.save_word2vec_format('')
# model.save('word2vec.model')


import numpy as np
import pickle
import os
import pandas as pd

def cut_word(sent):
    for words in sentences:


if __name__=="__main__":