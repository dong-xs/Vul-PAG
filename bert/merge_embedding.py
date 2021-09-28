# encoding:utf-8

'''
    本py文件的作用是：由于bert和tokenizer和spacy的tokenizer给出的单词序列是不一致的，其中bert由于采用wordpiece算法，
                   会将大的单词切分为小的单词，但我们最终的使用要用到spacy的tokenizer，因此必须要将bert切分的单词进行重新组装，
                   并将其嵌入求切分单词的均值来代替spacy的单词结果即可。。。
'''

import spacy
from bert_word_embedding import BertEmbedding
import torch

# spacy_nlp=spacy.load('en_core_web_sm')
spacy_nlp = spacy.load('en_core_web_md')
# 宿舍电脑用的是en_core_web_sm，实验室电脑用的是en_core_web_md

sentence = 'Arbitrary command execution via buffer overflow in Count.cgi (wwwcount) cgi-bin program.'
sentence1='Information from SSL-encrypted sessions via PKCS #1'
# 问题又来了，如果将上面这个句子进行解析，会有这种上下序列解析不一致的情况，例如：Count.cgi在spacy中没有解析出错，但在bert中会解析为Count . c ##gi这四个部分。

bert_embedding = BertEmbedding(sentence, 1, 0)


# 经tokenizer后发现，bert的token主要存在两种异常：
# （1）因为bert使用的是wordpiece，所以会使得将一个长的单词切分成几段
# （2）bert只会对原文本中出现过的子词进行一个embedding，因此如果已在文本中出现过而在其他组合词中出现的情况时，则不会再次出现该词的嵌入编码
# 因此，将embedding组装成spacy token的格式需要解决上述两种异常

def judge_starter(token):  # 用于判断某个字符是否以“##”开头，若是，则返回1，若不是，则返回0
    if token.startswith('##'):
        return 1
    else:
        return 0

# step1：首先将子词组合成长词：从以“##”开头这个部分出发，判断其下一个单词是否还是以“##”开头
def merge_embedding(embeds):     #该函数实现将以“##”开头的词进行合并,
    #输入：经bert_word_embedding运行出来的各嵌入层隐向量
    #输出：将以“##”开头的子词进行合并，并且返回其对应的token和embedding。

    bert_token = [key for key, value in embeds.items()][1:-1]  # 经bert的嵌入后的单词列表，类型为列表，元素长度为23

    bert_embedding = [value for key, value in embeds.items()][1:-1]  #去除掉头部的[cls]和尾部的[sep]，类型为列表,元素长度为23，
                                                                    # 其中每一个位置上是一个列表，列表中只有一个元素，且这个元素为一个tensor，
                                                                    #维度为：768。因此要embedding中每个词的值为：bert_embedding[i][0]

    start = []   #用于存放每个子词的开始位置
    end = []     #用于存放每个子词的结束位置
    for i in range(1, len(bert_token) - 1):
        if judge_starter(bert_token[i]) and not judge_starter(bert_token[i-1]) and not judge_starter(bert_token[i+1]):
            start.append(i-1)
            end.append(i)
        elif judge_starter(bert_token[i]) and not judge_starter(
                bert_token[i - 1]):  # 如果当前词以“##”开头，而前一个词不是以“##”开头，而上一个词为开始位置
            start.append(i - 1)
        elif judge_starter(bert_token[i]) and not judge_starter(
                bert_token[i + 1]):  # 如果当前词以“##”开头，而下一个词不是以“##”开头，则当前词为结束位置
            end.append(i)
        else:
            continue

    for s,e in zip(start,end):
        temp_token=bert_token[s]
        temp_embedding=bert_embedding[s][0]
        for i in range(s+1,e+1):
            temp_token+=bert_token[i][2:]
            bert_token[i]='-10000'
            temp_embedding+=bert_embedding[i][0]
            bert_embedding[i][0]=-10000
        temp_embedding=temp_embedding/(e-s+1)
        bert_embedding[s][0]=temp_embedding     #将嵌入部分进行合并，并且原“##”开头位置被代替为-10000
        bert_token[s]=temp_token    #得到合并后的token，原来以“##”开头的位置被代替为“-10000”
    index=[]   #用于存储
    for indexes in range(len(bert_embedding)):
        if isinstance(bert_embedding[indexes][0],int):  #只有替换为-10000的位置上为int，其余位置都为list类型
            index.append(indexes)

    bert_embedding=[bert_embedding[i] for i in range(len(bert_embedding)) if (i not in index)]

    while '-10000' in bert_token:
        bert_token.remove('-10000')

    return bert_token,bert_embedding

def split_index(str1,list1):
    for items in list1:
        if str1.startswith(items):
            return list1.index(items),items

# step2：将已出现词组合成spacy token中的词形式，最终输出肯定要按照
def spacy_bert_tokenizer_merge(bert_T,bert_E,content):
    docs = spacy_nlp(content)
    spacy_token = [str(token) for token in docs]  # 经spacy的tokenizer后的单词列表
    #在bert的token中出现的，一定会在spacy中出现，但在spacy中出现的不一定会参bert中出现，因此要找在spacy中而不在bert中的那些词
    diff_spacy=list(set(spacy_token)-set(bert_T))
    for diff_s in diff_spacy:
        index=[]
        embedding=torch.tensor(0)
        while diff_s is not None:
            temp_index,item=split_index(diff_s,bert_T)
            index.append(temp_index)
            diff_s=diff_s.split(item,1)[1:]
        for value in index:
            embedding+=bert_embedding[value][0]



new_token,new_embedding=merge_embedding(bert_embedding)
spacy_bert_tokenizer_merge(new_token,new_embedding,sentence)