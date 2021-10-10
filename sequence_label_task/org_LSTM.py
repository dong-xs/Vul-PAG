#encoding:utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(1)  # 设置cpu的随机数固定
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

START_TAG = '<START>'
STOP_TAG = '<STOP>'

EMBEDDING_DIM=128     #嵌入层的维度
HIDDEN_DIM=100        #隐藏层的维度
# epoch=2时，此时的标签准确率为：0.6879145357580627
# epoch=100时，此时的标签准确率为：0.755071453035032
# epoch=200时，此时的标签准确率为：0.7522936752572541

# EMBEDDING_DIM=128     #嵌入层的维度
# HIDDEN_DIM=50        #隐藏层的维度
# epoch=10时，此时的标签准确率为：0.7338673233938774

# EMBEDDING_DIM=256     #嵌入层的维度
# HIDDEN_DIM=200        #隐藏层的维度
# epoch=2时，此时的标签准确率为：0.6745642066582203
# epoch=100时，此时的标签准确率为：0.7302813829403706
# epoch=200时，此时的标签准确率为：0.732325510130507

# EMBEDDING_DIM=768     #嵌入层的维度
# HIDDEN_DIM=200        #隐藏层的维度
# epoch=2时，此时的标签准确率为：0.6180113730004223
# epoch=100时，此时的标签准确率为：0.7199031048175273
# epoch=200时，此时的标签准确率为：0.6912820730714955

# EMBEDDING_DIM = 768  # 嵌入层的维度
# HIDDEN_DIM = 100  # 隐藏层的维度

# epoch=2时，此时的标签准确率为：0.6225644680325083
# epoch=100时，此时的标签准确率为：0.7199031048175273
# epoch=200时，此时的标签准确率为：0.7054756110750595

def argmax(vec):
    _, idx = torch.max(vec, 1)
    # torch.max(input,dim)：
    # 输入：tensor，dim【说明：tensor就是一个输入值，dim表示维度，其中0表示每列的最大值，1表示每行的最大值】
    # 输出：两个tensor，第一个tensor表示每行/每列的最大值，第二个tensor表示最大值的索引。
    # 因此上面这一句的作用是返回一个输入tensor在行上的最大值，且将其索引存放于idx这样的一个tensor中
    return idx.item()  # tensor.item()表示获取该tensor的元素值
    # 综上所述，该函数的意义就是返回一个输入tensor在每一行上最大的索引值。


def prepare_sequence(seq, to_ix):
    # seq是一个输入的序列，
    # to_ix是一个标签对序列
    idxs = [to_ix[w] for w in seq]  # 这句代码的意思是将序列中的每个词转换到对应的的标签序列上，返回模式为一个列表
    return torch.tensor(idxs, dtype=torch.long)  # 将上述列表形式转换为tensor格式


def log_sum_exp(vec):  # 返回一个tensor中所有值与最大值的log sum exp
    max_score = vec[0, argmax(vec)]  # argmax(vec)将返回vec在行上的最大值，则max_score将会存放vec向量在行上的最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # tensor.view(1,-1)表示将原始tensor改变为1行，
    # tensor.expand(a,b)表示将原来的tensor复制扩展为a行b列的一个tensor，
    # 因此max_score_broadcast存放的是以vec最大值为元素的1行vec元素个数列的tensor
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        # 需要输入的参数包括：文本大小、标签与索引对应列表、嵌入层维度、隐层维度，这些参数是在初始化的时候需要输入的
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # tagset_size用于存储标签类别数

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # nn.Embedding(size,dim)
        # 输入：size表示文本一共有多少个词，dim表示为每个词设置的嵌入维度
        # 输出：若一个输入为m个词，则会输出一个[m,dim]的tensor，即每个位置上词将会用一个dim维的tensor代替。
        # 需要注意的是：这是的Embedding是pytorch自己定义后的嵌入框架，可以使用其他的嵌入方式如BERT、word2vec来代替

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # 定义一个单层的LSTM单元，
        # nn.LSTM(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)。
        # 输入：
        #   input_size:输入数据的特征维数，也就是词向量的维度；
        #   hidden_size: LSTM中隐层的维度；
        #   请注意，这是设定值为hidden_dim//2是因为，在使用双向LSTM时，前向和后向的最终输出维度为hidden_dim//2，
        #   将双向联合起来后其输出维度就是hidden_dim，这也便于后续的隐层向输出标签映射时的维度处理。但是隐藏层维度的意义是什么呢？
        #   num_layers:循环神经网络的层数，就是有多少个LSTM层的堆叠：
        #   如果设置多个网络层数，需要如何调整参数喃？
        #   bias：是否使用偏置，是一个boolen类型；
        #   batch_first:
        #   dropout:默认情况是0，
        #   bidirectional:是否使用双向LSTM，也是一个boolen类型。
        # 输出：主要包括两个部分，即output，（hn，cn）
        #       output：保存每个时间步的输出
        #       hn：句子最后一个单词的隐状态
        #       cn：句子最后一个单词的细胞状态

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # nn.Linear(in_feature,out_feature)用于设置网络中的全连接层
        # in_feature:表示网络的输入维度；
        # out_feature：表示网络的输出维度；
        # 最终输出就是一个target_size行target_size列的tensor

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # nn.Parameter()：这个函数可以理解为一个类型转换函数，就是将一个不可训练的tensor类型转换为可以训练的parameter类型；
        # torch.randn(a,b)：返回一个随机标量矩阵，大小为a行b列，且生成的值符从正态分布；
        # torch.rand(a,b):返回一个a行b列的随机标题矩阵，且内容符从均匀分布；
        # 因此该语句的意思就是生成一个tagset_size大小的标准矩阵后并将其转换为可训练的parameter形式，存于transitions变量中。

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # 将START_TAG标签对应索引所在行全部设置为-10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        # 将STOP_TAG标签对应索引所在列全部设置为-10000

        # self.hidden = self.init_hidden()

    def init_hidden(self):
        # 初始化的根本目的是随机生成输入时的h0，这里生成两个tensor的原因是使用BiLSTM，从前和从后面都需要初始化向量，但是每个向量的初始化为（2，1，hidden//2）是为啥喃？
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    # 初始化隐藏层，隐藏层为两个2行1列的tensor构成，每个位置上tensor值由一个hidden_dim大小的tensor构成

    # 这个_get_lstm_features函数就是用于获取LSTM的特征，如果要进行隐藏层的堆叠，可以在这儿进行处理。
    def _get_lstm_features(self, sentence):  # 该段用于获取句子的LSTM特征

        self.hidden = self.init_hidden()  # 首先初始化隐藏层参数
        # print(sentence)   #此处的sentence是每个句子的one-hot编码

        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # 然后通过嵌入层获得句子的嵌入表示，大小为x行1列,每个位置上的

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # 直接通过pytorch给定的LSMT函数获取上下文特征
        # 根据岳博士的建议，一般来说这儿的hidden层维度取embedding层维度开根号最比较合适的。
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  #

        lstm_feats = self.hidden2tag(lstm_out)      #返回每个词属于一个类别的概率值
        return lstm_feats

    def _forward_alg(self, feats):  # 使用前向算法来计算分区函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # torch.full(size,fill_value,out):是指返回一个值为fill_value、大小为size的张量
        # size:定义输出张量的形状；
        # fill_value:定义每个位置的填充值；
        # out：设定输出张量，一定设置为None；
        # 因此上式就是返回一个1行target_size列的张量，每个位置上的值为-10000.0

        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # 将初始化的参数第0行，START_TAG标签所在列的值设置为0

        forward_var = init_alphas  # 赋值给forward

        for feat in feats:  # 迭代遍历句子，针对第一个句子
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):  # 对每个分区句子的打分
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):  # 使用维特比算法处理输出端的解码问题，这个阶段我不需要进行处理
        backpointer = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointer.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointer):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        # print("the path_scores are:",path_score)
        # print("the best_path is",best_path)
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):  # 该函数用于构造模型的损失值，
        feats = self._get_lstm_features(sentence)  # 得到一个句子的LSMT判定结果
        forward_score = self._forward_alg(feats)  # 根据LSTM的判定概率得到对该判定结果的得分
        gold_score = self._score_sentence(feats, tags)  # 最优序列结果的得分
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)  # 得到LSTM的输出判定概率

        score, tag_seq = self._viterbi_decode(lstm_feats)  # viterbi接收LSTM的输出，并返回各个路径的评分以及最优的序列
        return score, tag_seq  # 这个就是整个模型的最终输出，每次输出有两个值，分别是最优得分及其对应的序列

# 训练阶段：加入自己标注的数据后的结果
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


train_data=dataset_get('../generate_data/train_data_zip.txt')
test_data=dataset_get('../generate_data/test_data_zip.txt')

tag_to_ix = {'B-VN': 0, 'I-VN': 1,
             'B-VV': 2, 'I-VV': 3,
             'B-VT': 4, 'I-VT': 5,
             'B-VRC': 6, 'I-VRC': 7,
             'B-VC': 8, 'I-VC': 9,
             'B-VP': 10, 'I-VP': 11,
             'B-VAT': 12, 'I-VAT': 13,
             'B-VAV': 14, 'I-VAV': 15,
             'B-VR': 16, 'I-VR': 17,
             'B-VF': 18, 'I-VF': 19,
             'O': 20, START_TAG: 21, STOP_TAG: 22}
# 添加自己标注数据部分到这儿为止
from torch.autograd import Variable
word_to_ix = {}     #获取所有word对应的索引值
for sentence, tags in train_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sentences, tages in test_data:
    for word in sentences:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)      #初始化一个模型
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)      #使用SGD进行优化

epoch_iter = 2

for epoch in range(epoch_iter):
    for sentence, tags in train_data:
        model.zero_grad()  # 每一步先清除梯度

        # 构造输入句子格式
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        # sentence_in, targets = Variable(data).to(device), Variable(
        #     target).to(device)
        # 对model执行前向运行
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 梯度更新与参数更新
        loss.backward()
        optimizer.step()
    print('Epoch [%d/%d] loss=%.4f' % (epoch + 1, epoch_iter, loss.item()))
    print("epoch: %d is finished!!!" % epoch)


def accuracy(list1, list2):
    count = 0
    for predict, orginal in zip(list1, list2):
        if predict == orginal:
            count += 1
    return count / len(list1)


with torch.no_grad():
    accuracy_score = []
    for item in test_data:
        precheck_sent = prepare_sequence(item[0], word_to_ix)
        predict_result = list(model(precheck_sent))
        accuracy_score.append(accuracy([tag_to_ix[value] for value in item[1]], predict_result[1]))
    print(accuracy_score)
    print(np.mean(accuracy_score))