# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from transformers import BertModel, BertTokenizer, BertConfig
from numpy import *
import re
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
import pandas as pd

model_name = 'bert-base-cased'
MODEL_PATH = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model_config = BertConfig.from_pretrained(model_name)
model_config.output_hidden_states = True
model = BertModel.from_pretrained(MODEL_PATH, config=model_config)

content=pd.read_csv('../sequence_label_task/labeled_24_31_space_new.csv')
description=content['description']

def dataset_get(filename):
    data = open(filename, 'r')
    content = data.readlines()
    data.close()

    indexes = [0]
    for i in range(len(content)):
        if content[i] == '\n':  # 找到每个位置为'\n'的索引
            indexes.append(i)

    indexes.append(-1)

    for value in range(2, len(indexes) - 1, 2):
        indexes[value] += 1

    sentence_label = []  # 用来存储每个分句后的结果
    for value in range(0, len(indexes) - 1, 2):
        sentence_label.append(content[indexes[value]:indexes[value + 1]])

    sent_length = len(sentence_label)
    # 接下来需要将里面的每个文本进行转换
    # 长度为句子长度，每个位置上的元素由两部分构成，即token序列和label序列构成
    train_data = [[] for index in range(sent_length)]

    for i in range(sent_length):
        temp_sent = []
        temp_label = []
        for value in sentence_label[i]:
            lists = value.strip().split('  ')
            temp_sent.append(lists[0])
            temp_label.append(lists[1])
        train_data[i] = (temp_sent, temp_label)
    return train_data

data=dataset_get('../generate_data/train_data_zip_spacesplit.txt')
label_list=[]     #存储所有标签
for item in data:
    label_list.append(item[1])

tag_to_ix = {'B-VN': 0, 'I-VN': 1,
             'B-VV': 2, 'I-VV': 3,
             'B-VT': 4, 'I-VT': 5,
             'B-VRC': 6, 'I-VRC': 7,
             'B-VP': 8, 'I-VP': 9,
             'B-VAT': 10, 'I-VAT': 11,
             'B-VAV': 12, 'I-VAV': 13,
             'B-VR': 14, 'I-VR': 15,
             'O': 16}

def get_label_tag(labellist):        #获取每个description中的所有标签，以及vat
    label_dict={}
    B_VAT=[]
    B_VR=[]
    B_VAV=[]
    for i in range(len(labellist)-1):
        start_index=0
        end_index=0
        if labellist[i].startswith('B') and (labellist[i+1].startswith('B') or labellist[i+1]=='O'):
            start_index=i
            end_index=i
            label_dict.setdefault(labellist[i],[]).append((start_index,end_index))

        elif labellist[i].startswith('B') and labellist[i+1].startswith('I'):
            start_index=i
            for j in range(i+1,len(labellist)-1):
                if labellist[j].startswith('I') and not labellist[j+1].startswith('I'):
                    end_index=j
                    label_dict.setdefault(labellist[i], []).append((start_index, end_index))
                    break

    if labellist[-1].startswith('B'):
        label_dict.setdefault(labellist[-1], []).append((len(labellist)-1,len(labellist)-1))

    #将B-VAV、B-VR、B-VAT的列表单独给出来
    if 'B-VAT' in label_dict.keys():
        B_VAT=label_dict['B-VAT']
    if 'B-VR' in label_dict.keys():
        B_VR=label_dict['B-VR']
    if 'B-VAV' in label_dict.keys():
        B_VAV=label_dict['B-VAV']
    return label_dict,B_VAT,B_VR,B_VAV   #返回每个标签的起始位置和终止位置

def spilt_sentence(text):
    re_tokens = [t for t in re.split(r'[\s+]', text.replace('\"', '')) if t]  # 判断是否为空时为了处理句尾有空格的情况
    tokens = []
    for token in re_tokens:
        if token[-1] not in [',', '.'] and '(' not in token and ')' not in token:
            tokens.append(token)
            continue
        temp = []
        if token[-1] in [',', '.']:
            temp.append(token[-1])
            token = token[:-1]
        token = token.replace('(', '( ').replace(')', ' )')
        token = [t for t in re.split(r'[\s+]', token) if t]
        tokens.extend(token)
        tokens.extend(temp)
    return tokens

def BertEmbedding(content):
    tokenized_text=spilt_sentence(content)
    indexes_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 将tokenizer后的结果全部转换为vocab_list中的索引
    segments_ids = [1] * len(tokenized_text)  # 用于存储每个句子的切分情况
    tokens_tensor = torch.tensor([indexes_tokens])  # 实现list向tensor的转换，便于与后续数据的数据结构一致
    segments_tensor = torch.tensor([segments_ids])

    with torch.no_grad():
        output = model(tokens_tensor,
                       segments_tensor)  # 当前model已经是封装好的，只需要将初始化的token和index的值传进去就ok了，这儿的segment设置为1，是因为每次给的batch为1，即一个句子的所有值设置为相同即可
        # 当前就是将一个句子的所有内容存放在outputs对象里了，依次分别是三个部分：last_hidden_state,pooler_output,hidden_state
        # 其中last_hidden_state中存储的是该句话共同构成的最终向量
        # hidden_state中存储的是12个layer中每一层的隐状态，且包含了每个词的嵌入表示，若要求每个词的最终嵌入，则需要将每个词的最后四层隐状态组合起来
    output1=output[0][0]       #仅返回bert最后一层的嵌入向量。。。输出维度为[seqlen,768]
    return output1

def attr_embedding(sent,taglist):
    word_embedding=BertEmbedding(sent)
    total_list,vat_list,vr_list,vav_list=get_label_tag(taglist)
    vat_embedding=[]
    vr_embedding = []
    vav_embedding = []

    if len(vat_list)==1:
        for item in vat_list:
            start=item[0]
            end=item[1]
            temp_embedding=0
            if start==end:
                vat_embedding.append(word_embedding[start])
            else:
                for i in range(start,end+1):
                    temp_embedding+=word_embedding[i]/np.linalg.norm(word_embedding[i])
                vat_embedding.append(temp_embedding/(end-start))
    elif len(vat_list)>1:
        for value in vat_list:
            temp_embedding = 0
            for item in value:
                start = item[0]
                end = item[1]
                if start == end:
                    vat_embedding.append(word_embedding[start])
                else:
                    for i in range(start, end + 1):
                        temp_embedding += word_embedding[i] / np.linalg.norm(word_embedding[i])
                    vat_embedding.append(temp_embedding / (end - start))


    for item in vr_list:
        start=item[0]
        end=item[1]
        temp_embedding=0
        if start==end:
            vat_embedding.append(word_embedding[start])
        else:
            for i in range(start,end+1):
                temp_embedding+=word_embedding[i]/np.linalg.norm(word_embedding[i])
            vr_embedding.append(temp_embedding/(end-start))

    for item in vav_list:
        start=item[0]
        end=item[1]
        temp_embedding=0
        if start==end:
            vat_embedding.append(word_embedding[start])
        else:
            for i in range(start,end+1):
                temp_embedding+=word_embedding[i]/np.linalg.norm(word_embedding[i])
            vav_embedding.append(temp_embedding/(end-start))

    return vat_embedding,vr_embedding,vav_embedding

vat_emb=[]
vr_emb=[]
vav_emb=[]
for sent,label in zip(description,label_list):
    temp_vat,temp_vr,temp_vav=attr_embedding(sent,label)
    vat_emb.append(temp_vat)
    vr_emb.append(temp_vr)
    vav_emb.append(temp_vav)
#当前得到的数据维度为（1390,seqlen,768）


kmeans=KMeans(n_clusters=3,random_state=1).fit(vat_emb)
# kmeans=KMeans(n_clusters=2,init='k-means++',random_state=1).fit(x)

y_hat=kmeans.labels_

score=calinski_harabasz_score(vat_emb,y_hat)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne=TSNE(n_components=2,init='random',random_state=0).fit(vat_emb)
df=pd.DataFrame(tsne.embedding_,index=vat_emb.index)
df1=df[df['labels']==0]
df2=df[df['labels']==1]
df3=df[df['labels']==2]
fig=plt.figure(figsize=(9,6))
plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD')
plt.show()