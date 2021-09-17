#encoding:utf-8
#bert的tensorflow安装可参照链接：https://cloud.tencent.com/developer/news/492066
#BERT启动命令：bert-serving-start -model_dir cased_L-12_H-768_A-12 -num_worker=1
#该模型参数的意思是12个layers，768的hidden-dim，12个head的attention
#在宿舍的电脑上跑不了bert程序，一跑就出错，只有尝试将bert放到服务器上跑试试看。。。

#推荐查看使用huggingface bert来进行微调：https://www.bilibili.com/video/BV1Dz4y1d7am

# from bert_serving.client import BertClient
# client=BertClient()
# vectors=client.encode(['vpn','shadonsocks','ss'])
#
# print(vectors)




'''
#本段为huggingface上的示例内容
import torch
from transformers import BertTokenizer,BertModel,BertForMaskedLM

import logging

logging.basicConfig(level=logging.INFO)

tokenizer=BertTokenizer.from_pretrained('bert-base-cased')

text='[CLS] Who was Jim ? [SEP] Jim was a puppeteer [SEP]'
tokenized_text=tokenizer.tokenize(text)   #用于将文本进行切分

indexed_tokens=tokenizer.convert_tokens_to_ids(tokenized_text)      #将text映射到词表中去，并返回其在词表中的索引位置
segments_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]      #这句话的意思就是将文本分句嘛，全为0的表示第一句，全为1的表示第二句
tokens_tensor=torch.tensor([indexed_tokens])
segments_tensors=torch.tensor([segments_ids])

model=BertModel.from_pretrained('bert-base-cased')
model.eval()

# 如果有GPU，则可以放到GPU上
# tokens_tensor=tokens_tensor.to('cuda')
# segments_tensors=segments_tensors.to('cuda')
# model.to('cuda')

with torch.no_grad():
    outputs=model(tokens_tensor,token_type_ids=segments_tensors)
    encoded_layers=outputs[0]

assert tuple(encoded_layers.shape)==(1,len(indexed_tokens),model.config.hidden_size)
print(encoded_layers)
'''


import torch
from transformers import BertModel,BertTokenizer
import logging

logging.basicConfig(level=logging.INFO)

tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
model=BertModel.from_pretrained('bert-base-cased')

file_path='../Dictionary_building/cwe_NER_set.txt'

f=open(file_path,'r')
content=f.readlines()[:5]
f.close()
text_list=''
for item in range(len(content)):
    text_list+='[CLS] '+content[item].strip()+ ' [SEP] '   #token_list存放了文本中的所有词

tokenized_text=tokenizer.tokenize((text_list))

'''
经过tokenizer后会出现很多以两个#号开头的内容，这也是需要处理的：出现的原因如下：
原来的单词被分成更小的子单词和字符。这些子单词前面的两个#号只是我们的tokenizer用来表示这个子单词或字符是一个更大单词的一部分，
并在其前面加上另一个子单词的方法。因此，例如，' ##bed ' token与' bed 'token是分开的，
当一个较大的单词中出现子单词bed时，使用第一种方法，当一个独立的token “thing you sleep on”出现时，使用第二种方法。
为什么会这样？这是因为BERT tokenizer 是用WordPiece模型创建的。这个模型使用贪心法创建了一个固定大小的词汇表
，其中包含单个字符、子单词和最适合我们的语言数据的单词。由于我们的BERT tokenizer模型的词汇量限制大小为30,000，
因此，用WordPiece模型生成一个包含所有英语字符的词汇表，再加上该模型所训练的英语语料库中发现的~30,000个最常见的单词和子单词。
'''

def smallist_word_index(lists):
    smallist_indexes=[]
    for i in range(len(lists)):
        if lists[i].strip().startswith('##'):
            smallist_indexes.append(i)
    return smallist_indexes

smallist_idx=smallist_word_index(tokenized_text)
print(smallist_idx)
for i in range(len(smallist_idx)-1):
    if smallist_idx[i+1]-smallist_idx[i]!=1:
        print(smallist_idx[i+1])
        print(smallist_idx[i])
        tokenized_text[smallist_idx[i-1]-1]=tokenized_text[smallist_idx[i]-1]+tokenized_text[smallist_idx[i]][2:]

print(tokenized_text)

# indexes_tokens=tokenizer.convert_tokens_to_ids(tokenized_text)
# for tup in zip(tokenized_text,indexes_tokens):
#     print(tup)
#
# segments_ids=[1]*len(tokenized_text)
#
# tokens_tensor=torch.tensor([indexes_tokens])
# segments_tensor=torch.tensor([segments_ids])
# model=BertModel.from_pretrained('bert-base-cased')
# model.eval()
#
# with torch.no_grad():
#     encoded_layers,encodes=model(tokens_tensor,segments_tensor)
#
# print(encodes)
# print('number of layers:',len(encoded_layers))
# print('number of batches:',len(encoded_layers[0]))
# print('number of tokens:',len(encoded_layers[0][0]))
# print('number of hidden units:',len(encoded_layers[0][0][0]))