#encoding:utf-8
#bert的tensorflow安装可参照链接：https://cloud.tencent.com/developer/news/492066
#BERT启动命令：bert-serving-start -model_dir cased_L-12_H-768_A-12 -num_worker=1
#该模型参数的意思是12个layers，768的hidden-dim，12个head的attention

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

'''
    该py文件的作用是输入一个句子，其按照nltk或spacy的tokenize后构成token，计算每个token的对就的词嵌入
    输入：一个句子
    输出：句子中每个token的嵌入
'''

import torch
from transformers import BertModel,BertTokenizer
import logging

logging.basicConfig(level=logging.INFO)

tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
model.eval()
sentence='The chpass command in OpenBSD allows a local user to gain root access through file descriptor leakage.'

def BertEmbedding(content,concate_lasted_4_layer,summed_lasted_4_layer):
    text_list='[CLS] '+content.strip()+ ' [SEP] '   #token_list存放了文本中的所有词
    tokenized_text=tokenizer.tokenize(text_list)

    indexes_tokens=tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids=[1]*len(tokenized_text)
    tokens_tensor=torch.tensor([indexes_tokens])
    segments_tensor=torch.tensor([segments_ids])

    with torch.no_grad():
        output=model(tokens_tensor,segments_tensor)
        #当前就是将一个句子的所有内容存放在outputs对象里了，依次分别是三个部分：last_hidden_state,pooler_output,hidden_state
        #其中last_hidden_state中存储的是该句话共同构成的最终向量
        #hidden_state中存储的是12个layer中每一层的隐状态，且包含了每个词的嵌入表示，若要求每个词的最终嵌入，则需要将每个词的最后四层隐状态组合起来

    outputs=output[2]    #只根据输出的最后一个参数来获取中间层的隐状态
    token_embedding=[]
    batch_i=0
    concate_lasted_4_layer_list=[]
    summed_lasted_4_layer_list=[]
    for token_i in range(len(tokenized_text)):   #对句子中的每个词进行遍历
        temp_concate_last_4=[]
        temp_summed_last_4=[]
        hidden_layers=[]
        for layer_i in range(len(outputs)):     #对12个层进行逐层遍历
            vec=outputs[layer_i][batch_i][token_i]    #获取每一层对同一个词的隐状态表示
            hidden_layers.append(vec)                 #将每一层的隐状态加入到hidden_layers列表中，则该列表中存放的一个token所有层的隐状态
        token_embedding.append(hidden_layers)        #token_embedding存放的是每一个token的每一层嵌入表示
        #有一个疑问，这儿的token_embedding和hidden_layers存在的意义是重复的，而hidden_list仅仅只是将每一层的数据转换为一个list。

        # 有两种方式来表示单词的最终向量：
        # （1）通过最后四层进行拼接来获得最终的嵌入向量表示，在使用该种情况表示最后的嵌入向量时，会存在维度过高而增加计算量
        temp_concate_last_4=[torch.cat((layer[-1],layer[-2],layer[-3],layer[-4]),0) for layer in token_embedding]
        concate_lasted_4_layer_list.append(temp_concate_last_4)
        # （2）通过将最后四层进行平均来获得最终的嵌入向量表示
        temp_summed_last_4=[torch.sum(torch.stack(layer)[-4:],0) for layer in token_embedding]
        summed_lasted_4_layer_list.append(temp_summed_last_4)
        #到目前为止，可以输出每个token的嵌入向量

    if concate_lasted_4_layer:
        return concate_lasted_4_layer_list,tokenized_text
    if summed_lasted_4_layer:
        return summed_lasted_4_layer_list,tokenized_text

embedding,tokens=BertEmbedding(sentence,1,0)
for word,vec in zip(tokens,embedding):
    print(word,vec)
'''
经过tokenizer后会出现很多以两个#号开头的内容，这也是需要处理的：出现的原因如下：
    原来的单词被分成更小的子单词和字符。这些子单词前面的两个#号只是我们的tokenizer用来表示这个子单词或字符是一个更大单词的一部分，
并在其前面加上另一个子单词的方法。因此，例如，'##bed' token与'bed 'token是分开的，当一个较大的单词中出现子单词bed时，
使用第一种方法，当一个独立的token “thing you sleep on”出现时，使用第二种方法。
为什么会这样？这是因为BERT tokenizer 是用WordPiece模型创建的。这个模型使用贪心法创建了一个固定大小的词汇表，
其中包含单个字符、子单词和最适合我们的语言数据的单词。由于我们的BERT tokenizer模型的词汇量限制大小为30,000，
因此，用WordPiece模型生成一个包含所有英语字符的词汇表，再加上该模型所训练的英语语料库中发现的~30,000个最常见的单词和子单词。

我们首先获取一下当前bert中预训练模型中词汇表的内容，因为存在“##XX”的情况，
需要看一下是不是对所有的相同单词都是相同处理的，如果都是一致处理的，则我们就不需要进行合并调整；
如果对同一单词是进行的不同处理，则需要将“##XX”的单词进行合并。
经选择上述测试文档中的[6:11]部分看出来，其tokenizer对每个单词的处理都是一致的，因此不用进行调整。
with open('bert_vocab_list.txt','w',encoding='utf-8') as f:
    for item in list(tokenizer.vocab.keys()):
        f.write(str(item)+'\n')
f.close()
'''
