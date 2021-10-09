import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bert.merge_embedding import spacy_bert_tokenizer_merge

torch.manual_seed(1)  # ����cpu��������̶�

START_TAG = '<START>'
STOP_TAG = '<STOP>'

# EMBEDDING_DIM=128     #Ƕ����ά��
# HIDDEN_DIM=100        #���ز��ά��
# epoch=2ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.6879145357580627
# epoch=100ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.755071453035032
# epoch=200ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.7522936752572541


# EMBEDDING_DIM=256     #Ƕ����ά��
# HIDDEN_DIM=200        #���ز��ά��
# epoch=2ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.6745642066582203
# epoch=100ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.7302813829403706
# epoch=200ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.732325510130507

# EMBEDDING_DIM=768     #Ƕ����ά��
# HIDDEN_DIM=200        #���ز��ά��
# epoch=2ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.6180113730004223
# epoch=100ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.7199031048175273
# epoch=200ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.6912820730714955

EMBEDDING_DIM = 768  # Ƕ����ά��
HIDDEN_DIM = 100  # ���ز��ά��


# epoch=2ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.6225644680325083
# epoch=100ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.7199031048175273
# epoch=200ʱ����ʱ�ı�ǩ׼ȷ��Ϊ��0.7054756110750595

def argmax(vec):
    _, idx = torch.max(vec, 1)
    # torch.max(input,dim)��
    # ���룺tensor��dim��˵����tensor����һ������ֵ��dim��ʾά�ȣ�����0��ʾÿ�е����ֵ��1��ʾÿ�е����ֵ��
    # ���������tensor����һ��tensor��ʾÿ��/ÿ�е����ֵ���ڶ���tensor��ʾ���ֵ��������
    # ���������һ��������Ƿ���һ������tensor�����ϵ����ֵ���ҽ������������idx������һ��tensor��
    return idx.item()  # tensor.item()��ʾ��ȡ��tensor��Ԫ��ֵ
    # �����������ú�����������Ƿ���һ������tensor��һ������������ֵ��


def prepare_sequence(seq, to_ix):
    # seq��һ����������У�
    # to_ix��һ����ǩ������
    idxs = [to_ix[w] for w in seq]  # ���������˼�ǽ������е�ÿ����ת������Ӧ�ĵı�ǩ�����ϣ�����ģʽΪһ���б�
    return torch.tensor(idxs, dtype=torch.long)  # �������б���ʽת��Ϊtensor��ʽ


def log_sum_exp(vec):  # ����һ��tensor������ֵ�����ֵ��log sum exp
    max_score = vec[0, argmax(vec)]  # argmax(vec)������vec�����ϵ����ֵ����max_score������vec���������ϵ����ֵ
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # tensor.view(1,-1)��ʾ��ԭʼtensor�ı�Ϊ1�У�
    # tensor.expand(a,b)��ʾ��ԭ����tensor������չΪa��b�е�һ��tensor��
    # ���max_score_broadcast��ŵ�����vec���ֵΪԪ�ص�1��vecԪ�ظ����е�tensor
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        # ��Ҫ����Ĳ����������ı���С����ǩ��������Ӧ�б�Ƕ���ά�ȡ�����ά��
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # tagset_size���ڴ洢��ǩ�����

        self.word_embeds=spacy_bert_tokenizer_merge()
        # nn.Embedding(size,dim)
        # ���룺size��ʾ�ı�һ���ж��ٸ��ʣ�dim��ʾΪÿ�������õ�Ƕ��ά��
        # �������һ������Ϊ[m,n]��������һ��[m,n,dim]��tensor����ÿ��λ���ϴʽ�����һ��dimά��tensor���档
        # ��Ҫע����ǣ����ǵ�Embedding��pytorch�Լ�������Ƕ���ܣ�����ʹ��������Ƕ�뷽ʽ��BERT��word2vec������

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # ����һ�������LSTM��Ԫ��
        # nn.LSTM(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)��
        # ���룺
        #   input_size:�������ݵ�����ά����Ҳ���Ǵ�������ά�ȣ�
        #   hidden_size: LSTM�������ά�ȣ�
        #   ��ע�⣬�����趨ֵΪhidden_dim//2����Ϊ����ʹ��˫��LSTMʱ��ǰ��ͺ�����������ά��Ϊhidden_dim//2��
        #   ��˫�����������������ά�Ⱦ���hidden_dim����Ҳ���ں����������������ǩӳ��ʱ��ά�ȴ����������ز�ά�ȵ�������ʲô�أ�
        #   num_layers:ѭ��������Ĳ����������ж��ٸ�LSTM��Ķѵ���
        #   ������ö�������������Ҫ��ε�������ૣ�
        #   bias���Ƿ�ʹ��ƫ�ã���һ��boolen���ͣ�
        #   batch_first:
        #   dropout:Ĭ�������0��
        #   bidirectional:�Ƿ�ʹ��˫��LSTM��Ҳ��һ��boolen���͡�
        # �������Ҫ�����������֣���output����hn��cn��
        #       output������ÿ��ʱ�䲽�����
        #       hn���������һ�����ʵ���״̬
        #       cn���������һ�����ʵ�ϸ��״̬

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # nn.Linear(in_feature,out_feature)�������������е�ȫ���Ӳ�
        # in_feature:��ʾ���������ά�ȣ�
        # out_feature����ʾ��������ά�ȣ�
        # �����������һ��target_size��target_size�е�tensor

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # nn.Parameter()����������������Ϊһ������ת�����������ǽ�һ������ѵ����tensor����ת��Ϊ����ѵ����parameter���ͣ�
        # torch.randn(a,b)������һ������������󣬴�СΪa��b�У������ɵ�ֵ������̬�ֲ���
        # torch.rand(a,b):����һ��a��b�е����������������ݷ��Ӿ��ȷֲ���
        # ��˸�������˼��������һ��tagset_size��С�ı�׼����󲢽���ת��Ϊ��ѵ����parameter��ʽ������transitions�����С�

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # ��START_TAG��ǩ��Ӧ����������ȫ������Ϊ-10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        # ��STOP_TAG��ǩ��Ӧ����������ȫ������Ϊ-10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # ��ʼ���ĸ���Ŀ���������������ʱ��h0��������������tensor��ԭ����ʹ��BiLSTM����ǰ�ʹӺ��涼��Ҫ��ʼ������������ÿ�������ĳ�ʼ��Ϊ��2��1��hidden//2����Ϊɶૣ�
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    # ��ʼ�����ز㣬���ز�Ϊ����2��1�е�tensor���ɣ�ÿ��λ����tensorֵ��һ��hidden_dim��С��tensor����

    # ���_get_lstm_features�����������ڻ�ȡLSTM�����������Ҫ�������ز�Ķѵ���������������д���
    def _get_lstm_features(self, sentence):  # �ö����ڻ�ȡ���ӵ�LSTM����
        self.hidden = self.init_hidden()  # ���ȳ�ʼ�����ز����
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # Ȼ��ͨ��Ƕ����þ��ӵ�Ƕ���ʾ����СΪx��1��
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # ֱ��ͨ��pytorch������LSMT������ȡ����������
        # ��������ʿ�Ľ��飬һ����˵�����hidden��ά��ȡembedding��ά�ȿ�������ȽϺ��ʵġ�
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  #

        lstm_feats = self.hidden2tag(lstm_out)
        # print("the value of lstm_feats is:",lstm_feats)
        return lstm_feats

    def _forward_alg(self, feats):  # ʹ��ǰ���㷨�������������
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # torch.full(size,fill_value,out):��ָ����һ��ֵΪfill_value����СΪsize������
        # size:���������������״��
        # fill_value:����ÿ��λ�õ����ֵ��
        # out���趨���������һ������ΪNone��
        # �����ʽ���Ƿ���һ��1��target_size�е�������ÿ��λ���ϵ�ֵΪ-10000.0

        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # ����ʼ���Ĳ�����0�У�START_TAG��ǩ�����е�ֵ����Ϊ0

        forward_var = init_alphas  # ��ֵ��forward

        for feat in feats:  # �����������ӣ���Ե�һ������
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # print("the _forward_alg value is:",alpha)
        return alpha

    def _score_sentence(self, feats, tags):  # ��ÿ���������ӵĴ��
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        # print('the score of sentence is :',score)
        return score

    def _viterbi_decode(self, feats):  # ʹ��ά�ر��㷨��������˵Ľ������⣬����׶��Ҳ���Ҫ���д���
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

    def neg_log_likelihood(self, sentence, tags):  # �ú������ڹ���ģ�͵���ʧֵ��
        feats = self._get_lstm_features(sentence)  # �õ�һ�����ӵ�LSMT�ж����
        forward_score = self._forward_alg(feats)  # ����LSTM���ж����ʵõ��Ը��ж�����ĵ÷�
        gold_score = self._score_sentence(feats, tags)  # �������н���ĵ÷�
        # print("the diff of forward_score and gold_score is :",forward_score-gold_score)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)  # �õ�LSTM������ж�����

        score, tag_seq = self._viterbi_decode(lstm_feats)  # viterbi����LSTM������������ظ���·���������Լ����ŵ�����
        return score, tag_seq  # �����������ģ�͵����������ÿ�����������ֵ���ֱ������ŵ÷ּ����Ӧ������

# ѵ���׶Σ������Լ���ע�����ݺ�Ľ��
def dataset_get(filename):
    data = open(filename, 'r')
    content = data.readlines()
    data.close()

    indexes = [0]
    for i in range(len(content)):
        if content[i] == '\n':  # �ҵ�ÿ��λ��Ϊ'\n'������
            indexes.append(i)

    indexes.append(-1)

    for value in range(2, len(indexes) - 1, 2):
        indexes[value] += 1

    sentence_label = []  # �����洢ÿ���־��Ľ��
    for value in range(0, len(indexes) - 1, 2):
        sentence_label.append(content[indexes[value]:indexes[value + 1]])

    sent_length = len(sentence_label)
    # ��������Ҫ�������ÿ���ı�����ת��
    # ����Ϊ���ӳ��ȣ�ÿ��λ���ϵ�Ԫ���������ֹ��ɣ���token���к�label���й���
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


train_data = dataset_get('test_data.txt')
test_data = dataset_get('lqd_label_result_test.txt')


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
# ����Լ���ע���ݲ��ֵ����Ϊֹ

word_to_ix = {}
for sentence, tags in train_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sentences, tages in test_data:
    for word in sentences:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

epoch_iter = 200

for epoch in range(epoch_iter):
    for sentence, tags in train_data:
        model.zero_grad()  # ÿһ��������ݶ�

        # ����������Ӹ�ʽ
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # ��modelִ��ǰ������
        loss = model.neg_log_likelihood(sentence_in, targets)

        # �ݶȸ������������
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
