#encoding:utf-8

import spacy
from bert_word_embedding import BertEmbedding

spayc_nlp=spacy.load('en_core_web_md')

sentence='The chpass command in OpenBSD allows a local user to gain root access through file descriptor leakage.'
bert_embedding=BertEmbedding(sentence,1,0)

#需要将sentence按照spacy的tokenizer后与当前BERT的tokenizer进行比较，将bert中的embedding进行合并
import spacy
spacy_nlp=spacy.load('en_core_web_md')
def merge_embedding(embeds,content):
    docs=spacy_nlp(content)
    spacy_token=[str(token) for token in docs]   #经spacy的tokenizer后的单词列表
    bert_token=[key for key,value in embeds.items()][1:-2]   #经bert的嵌入后的单词列表
    bert_embedding=[value for key,value in embeds.items()][1:-2]
    result_embedding=[]

    for i in range(len(bert_token)):
        temp_embedding=bert_embedding[i]
        if bert_token[i] in spacy_token:
            result_embedding.append(temp_embedding)
        elif bert_token[i] not in spacy_token and bert_token[i+1].startswith('##'):

    abs_index=[]   #用于存储那些没有切割的单词索引位置
    org_index=[i for i in range(len(bert_token))]
    start=0
    #接下来需要对spacy_token和bert_token的各个单词进行比较，然后返回其BERT嵌入
    for s_item in range(len(spacy_token)):
        if spacy_token[s_item] in bert_token:
            abs_index.append(bert_token.index(spacy_token[s_item],start))
            start+=1
        else:
            pass
    diff_index=list(set(org_index)-set(abs_index))     #用于存储bert token中切分后的单词的索引值
    return spacy_token,bert_token,abs_index,diff_index

s,b,i,d=merge_embedding(bert_embedding,sentence)
print(s)
print(b)
print(i)
print(d)