#encoding:utf-8


import torch
from torch import nn as nn

EMBEDDING_DIM=768     #嵌入层的维度
HIDDEN_DIM=100        #隐藏层的维度

from transformers import BertModel,BertTokenizer
bertModel=BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
bertModel.eval()

class bert_lstm_crf(nn.Module):
    def __init__(self,embed_dim,hidden_dim,tag_to_ix):
        super(bert_lstm_crf, self).__init__()
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.tag_to_ix=tag_to_ix
        self.target_size=len(tag_to_ix)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.lstm=nn.LSTM(embed_dim,hidden_dim//2,num_layers=1,bidirectional=False)

        self.hidden2tag=nn.Linear(hidden_dim,self.target_size)

    def init_hidden(self):
        return (torch.randn(2,1,self.embed_dim),torch.randn(2,1,self.embed_dim))

    def BertEmbedding(contents):
        # 参数说明：
        #   content：表示输入的文本内容
        #   concate_lasted_4_layer：boolean类型，即选择以最后4层拼接的形式返回结果，默认为0
        #   summed_lasted_4_layer：boolean类型，即选择以最后4的拼接形式返回结果，默认为1
        text_list = '[CLS] ' + contents.strip() + ' [SEP] '  # token_list存放了文本中的所有词
        tokenized_text = tokenizer.tokenize(text_list)  # 此处的tokenizer会将一个完整的单词转换为多个小部分单词，
        # 在这里需要按照spacy的tokenizer格式来处理

        indexes_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 将tokenizer后的结果全部转换为vocab_list中的索引
        segments_ids = [1] * len(tokenized_text)  # 用于存储每个句子的切分情况
        tokens_tensor = torch.tensor([indexes_tokens])  # 实现list向tensor的转换，便于与后续数据的数据结构一致
        segments_tensor = torch.tensor([segments_ids])

        with torch.no_grad():
            output = bertModel(tokens_tensor,
                           segments_tensor)  # 当前model已经是封装好的，只需要将初始化的token和index的值传进去就ok了，这儿的segment设置为1，是因为每次给的batch为1，即一个句子的所有值设置为相同即可
            # 当前就是将一个句子的所有内容存放在outputs对象里了，依次分别是三个部分：last_hidden_state,pooler_output,hidden_state
            # 其中last_hidden_state中存储的是该句话共同构成的最终向量
            # hidden_state中存储的是12个layer中每一层的隐状态，且包含了每个词的嵌入表示，若要求每个词的最终嵌入，则需要将每个词的最后四层隐状态组合起来

        outputs = output[2]  # 只根据输出的最后一个参数来获取中间层的隐状态

        batch_i = 0  # 此处设置batch_i=0是因为每次送入bert的结果都是一个句子
        concate_lasted_4_layer_list = {}
        summed_lasted_4_layer_list = {}
        for token_i in range(len(tokenized_text)):  # 对句子中的每个词进行遍历
            token_embedding = []  # 记录一个词所有层的隐状态
            hidden_layers = []
            for layer_i in range(len(outputs)):  # 对12个层进行逐层遍历
                vec = outputs[layer_i][batch_i][token_i]  # 获取每一层对同一个词的隐状态表示，维度为[1,768]
                hidden_layers.append(vec)  # 将每一层的隐状态加入到hidden_layers列表中，则该列表中存放的一个token所有层的隐状态，维度为[13,768]
            token_embedding.append(hidden_layers)  # token_embedding存放的是所有token的所有层嵌入表示，维度为[1,13,768]
            # 有一个疑问，这儿的token_embedding和hidden_layers存在的意义是重复的，而hidden_list仅仅只是将每一层的数据转换为一个list。

            # 有两种方式来表示单词的最终向量：
            # （1）通过最后四层进行拼接来获得最终的嵌入向量表示，在使用该种情况表示最后的嵌入向量时，会存在维度过高而增加计算量
            temp_concate_last_4 = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                                   token_embedding]
            concate_lasted_4_layer_list[tokenized_text[token_i]] = temp_concate_last_4
            # （2）通过将最后四层进行求和来获得最终的嵌入向量表示,此处返回的值维度为输入维度的正常值。
            temp_summed_last_4 = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embedding]
            summed_lasted_4_layer_list[tokenized_text[token_i]] = temp_summed_last_4
            # 到目前为止，返回每个token的嵌入向量，以字典形式返回。

        return summed_lasted_4_layer_list  # 返回每个token的维度为768

    def _get_lstm_feature(self,sentence):
        self.hidden=self.init_hidden()

        embed=BertModel(sentence)

        lstm_out,last_hidden=self.lstm(embed,self.hidden)

        lstm_feat=self.hidden2tag(lstm_out)

        return lstm_feat

    def forward(self,sentence):
        feats=self._get_lstm_feature(sentence)
        return feats

tag2ix = {'B-VN': 0, 'I-VN': 1,
             'B-VV': 2, 'I-VV': 3,
             'B-VT': 4, 'I-VT': 5,
             'B-VRC': 6, 'I-VRC': 7,
             'B-VC': 8, 'I-VC': 9,
             'B-VP': 10, 'I-VP': 11,
             'B-VAT': 12, 'I-VAT': 13,
             'B-VAV': 14, 'I-VAV': 15,
             'B-VR': 16, 'I-VR': 17,
             'B-VF': 18, 'I-VF': 19,
             'O': 20}

from torch import optim
model=bert_lstm_crf(embed_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,tag_to_ix=tag2ix)
optmizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-4)

