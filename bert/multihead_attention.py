# encoding:utf-8

import torch
import math
import torch.functional as F
import torch.nn as nn



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

        '''
            按照视频的解释：https://www.bilibili.com/video/BV1P4411F77q
                Q=Linear(Xembedding)=Xembedding*WQ     即两个矩阵之间求点积，其中WQ,WK,WV三个矩阵的维度均为(embedding,embedding)
                K=Linear(Xembedding)=Xembedding*WK
                V=Linear(Xembedding)=Xembedding*WV
                Q,K,V的维度为[batch-size,seq.length,embed.dim]
                经过多头后，则Q,K,V的维度变为[batch-size,seq.length,h,embed.dim/h]
        '''

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
