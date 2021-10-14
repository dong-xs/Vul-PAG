# encoding:utf-8

'''
    该py文件的作用是输入一个句子，其按照nltk或spacy的tokenize后构成token，计算每个token的对就的词嵌入
    输入：一个句子
    输出：句子中每个token的嵌入
'''

import torch
from transformers import BertModel, BertTokenizer, BertConfig
import spacy
from numpy import *

model_name = 'bert-base-cased'
MODEL_PATH = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name,do_lower_case=False)
model_config = BertConfig.from_pretrained(model_name)
model_config.output_hidden_states = True
model = BertModel.from_pretrained(MODEL_PATH, config=model_config)

sentence = 'Management information base (MIB) for a 3Com SuperStack II hub running software version 2.10 contains an object identifier (.1.3.6.1.4.1.43.10.4.2) that is accessible by a read-only community string, but lists the entire table of community strings, which could allow attackers to conduct unauthorized activities.'


# 问题又来了，如果将上面这个句子进行解析，会有这种上下序列解析不一致的情况，例如：Count.cgi在spacy中没有解析出错，但在bert中会解析为Count . c ##gi这四个部分。

def BertEmbedding(content):
    tokenized_text = tokenizer.tokenize(content,)  # 此处的tokenizer会将一个完整的单词转换为多个小部分单词，
    # 在这里需要按照spacy的tokenizer格式来处理
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

    outputs = output[2]  # 只根据输出的最后一个参数来获取中间层的隐状态

    batch_i = 0  # 此处设置batch_i=0是因为每次送入bert的结果都是一个句子
    summed_lasted_4_layer_list = {}
    for token_i in range(len(tokenized_text)):  # 对句子中的每个词进行遍历
        token_embedding = []  # 记录一个词所有层的隐状态
        hidden_layers = []
        for layer_i in range(len(outputs)):  # 对12个层进行逐层遍历
            vec = outputs[layer_i][batch_i][token_i]  # 获取每一层对同一个词的隐状态表示，维度为[1,768]
            hidden_layers.append(vec)  # 将每一层的隐状态加入到hidden_layers列表中，则该列表中存放的一个token所有层的隐状态，维度为[13,768]
        token_embedding.append(hidden_layers)  # token_embedding存放的是所有token的所有层嵌入表示，维度为[1,13,768]
        # 有一个疑问，这儿的token_embedding和hidden_layers存在的意义是重复的，而hidden_list仅仅只是将每一层的数据转换为一个list。

        temp_summed_last_4 = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embedding]
        summed_lasted_4_layer_list[tokenized_text[token_i]] = temp_summed_last_4
        # 到目前为止，返回每个token的嵌入向量，以字典形式返回。
    return summed_lasted_4_layer_list  # 返回每个token的维度为768,构成方式为：token：embedding


# 说明：此处返回的是经过tokenizer后长度个数的，也就是说这是经过wordpiece后的值，还是需要将一定的值进行组合起来，构成一个完整的词的嵌入表示。，例如当前这句话给出来的词长度为24，

spacy_nlp = spacy.load('en_core_web_sm')


# spacy_nlp = spacy.load('en_core_web_md')


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
def merge_embedding(sentence):  # 该函数实现将以“##”开头的词进行合并,
    # 输入：经bert_word_embedding运行出来的各嵌入层隐向量
    # 输出：将以“##”开头的子词进行合并，并且返回其对应的token和embedding。

    embeds = BertEmbedding(sentence)  # bert_embedding的类型是一个字典类型，key值为一个subtoken，value为其tensor

    bert_token = [key for key, value in embeds.items()]  # 经bert的嵌入后的单词列表，类型为列表，元素长度为23
    bert_token1 = [key for key, value in embeds.items()]

    bert_embeddings = [value for key, value in embeds.items()]  # 去除掉头部的[cls]和尾部的[sep]，类型为列表,元素长度为23，
    bert_embeddings1 = [value for key, value in embeds.items()]
    # 其中每一个位置上是一个列表，列表中只有一个元素，且这个元素为一个tensor，
    # 维度为：768。因此要embedding中每个词的值为：bert_embedding[i][0]

    start = []  # 用于存放每个子词的开始位置
    end = []  # 用于存放每个子词的结束位置
    for i in range(1, len(bert_token) - 1):
        if judge_starter(bert_token[i]) and not judge_starter(bert_token[i - 1]) and not judge_starter(
                bert_token[i + 1]):
            start.append(i - 1)
            end.append(i)
        elif judge_starter(bert_token[i]) and not judge_starter(
                bert_token[i - 1]):  # 如果当前词以“##”开头，而前一个词不是以“##”开头，而上一个词为开始位置
            start.append(i - 1)
        elif judge_starter(bert_token[i]) and not judge_starter(
                bert_token[i + 1]):  # 如果当前词以“##”开头，而下一个词不是以“##”开头，则当前词为结束位置
            end.append(i)
        else:
            continue

    for s, e in zip(start, end):
        temp_token = bert_token[s]
        temp_embedding = bert_embeddings[s][0]
        for j in range(s + 1, e + 1):
            temp_token += bert_token[j][2:]
            bert_token[j] = '-10000'
            temp_embedding += bert_embeddings[j][0]
            bert_embeddings[j] = -10000
        temp_embedding = temp_embedding / (e - s + 1)
        bert_embeddings[s] = temp_embedding  # 将嵌入部分进行合并，并且原“##”开头位置被代替为-10000
        bert_token[s] = temp_token  # 得到合并后的token，原来以“##”开头的位置被代替为“-10000”

    index = []  # 用于存储为-10000位置的索引
    for indexes in range(len(bert_embeddings)):
        if isinstance(bert_embeddings[indexes], int):  # 只有替换为-10000的位置上为int，其余位置都为list类型
            index.append(indexes)

    bert_embeddings = [bert_embeddings[i] for i in range(len(bert_embeddings)) if (i not in index)]

    while '-10000' in bert_token:
        bert_token.remove('-10000')

    new_bert_tokens = list(set(bert_token) | set(bert_token1))
    new_bert_embedding = []

    for tokens in new_bert_tokens:
        if tokens in bert_token:
            index = bert_token.index(tokens)
            new_bert_embedding.append(bert_embeddings[index])
        elif tokens in bert_token1:
            index = bert_token1.index(tokens)
            new_bert_embedding.append(bert_embeddings1[index])

    return new_bert_tokens, new_bert_embedding  # 此时返回的是将“##”合并后的结果，但也会存在原句子中未分割的情况，需要在这儿将最开始的bert-token和bert-embedding当作值加进来


def split_index(str1, list1):
    temp_index = []
    temp_item = []
    for items in list1:
        if str1.startswith(items) and items != '':
            temp_item.append(items)
            temp_index.append(list1.index(items))

    max_len_item = max(temp_item, key=len, default='')

    max_len_index = list1.index(max_len_item)
    return max_len_index, max_len_item


# step2：将已出现词组合成spacy token中的词形式，最终输出按照spacy token的顺序依次成为字典输出
def spacy_bert_tokenizer_merge(content):
    bert_T, bert_E = merge_embedding(content)
    docs = spacy_nlp(content)
    spacy_token = [str(token) for token in docs]  # 经spacy的tokenizer后的单词列表
    # 在bert的token中出现的，一定会在spacy中出现，但在spacy中出现的不一定会参bert中出现，因此要找在spacy中而不在bert中的那些词

    final_embed = [0 for i in range(len(spacy_token))]

    # 先处理两者的差集
    diff_spacy = list(set(spacy_token) - set(bert_T))
    diff_result = {}  # 用于存储所有在diff_spacy中的结果

    for diff_s in diff_spacy:  # 遍历diff_spacy中的每一个元素

        copy_diff_s = diff_s
        index = []
        embedding = torch.zeros(1, 768)  # 生成一个一行768列的全0 tensor
        while diff_s != '':  # 当每一个元素不为空字符串时，因为split在最后一次切分后必定会出现一个空字符串''，以此作为循环结束条件
            temp_index, item = split_index(diff_s.strip(), bert_T)  # 将每个diff_s都在bert-T找到对应的词以及索引位置
            index.append(temp_index)  # 暂存每个子词的索引位置
            diff_s = diff_s.split(item, 1)[-1]  # 用于暂存split后的结果,split后会形成一个列表，因为设置了只分割一次，每次分割后用后面的一部分作为新的diff_s

        for value in index:  # 接下来需要将各个索引位置的值进行相加
            embedding += bert_E[value][0]
        embedding = embedding / len(index)  # 这是输出的是一个词的embedding
        diff_result[copy_diff_s] = embedding

    # 对spacy_token进行遍历以找到每个词对应的embedding
    for items in range(len(spacy_token)):
        if spacy_token[items] in bert_T:
            final_embed[items] = bert_E[bert_T.index(spacy_token[items])]
        elif spacy_token[items] in list(diff_result.keys()):
            final_embed[items] = diff_result[spacy_token[items]]
    return final_embed
