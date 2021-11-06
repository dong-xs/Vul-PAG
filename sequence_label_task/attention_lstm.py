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

EMBEDDING_DIM=300     #嵌入层的维度
HIDDEN_DIM=120        #隐藏层的维度

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):  # 返回一个tensor中所有值与最大值的log sum exp
    max_score = vec[0, argmax(vec)]  # argmax(vec)将返回vec在行上的最大值，则max_score将会存放vec向量在行上的最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        '''
        :param hid_dim:隐藏层维度，即为上一个嵌入层的输出
        :param n_heads: 多头数目，一般为12或24
        :param dropout:dropout主要是为了过拟合而设置的,即在前向传播时,让某个激活值以一定的概率p停止工作,使模型的泛化性更强
        :param device:确定是否需要使用GPU
        '''
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0  # 设置断言，因为multihead_attention会把每个embedding拆分成n_heads份，
        # 每一次拿每一份的相应部分作self_attention，最后再将所有结果组合起来即可，这也是narrow self-attention机制
        self.w_q = nn.Linear(hid_dim, hid_dim)  # 即将w_q赋值为一个函数，其作用是输入一个hid_dim维的数据，输出一个hid_dim维的数据
        #nn.Linear()函数的作用就是将实现两个向量的映射，在此过程中会随机分配一个qkv，而随着训练过程来更新qkv。
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) // self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contigous()
        x = x.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x, attention

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # tagset_size用于存储标签类别数

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.attention = SelfAttention(hidden_dim, n_heads=12, dropout=0.01, device=None)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)


        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))


    def _get_lstm_features(self, sentence):  # 该段用于获取句子的LSTM特征

        self.hidden = self.init_hidden()  # 首先初始化隐藏层参数
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # 直接通过pytorch给定的LSMT函数获取上下文特征
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  #

        _, attention_score = self.attention(lstm_out, lstm_out, lstm_out)

        lstm_feats=self.hidden2tag(attention_score)
        return lstm_feats
        # lstm_feats = self.hidden2tag(lstm_out)      #返回每个词属于一个类别的概率值
        # return lstm_feats,attention_score

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