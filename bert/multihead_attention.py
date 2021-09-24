#encoding:utf-8

import torch
import math
import torch.functional as F
import torch.nn as nn

def attention(query,key,value,mask=None,dropout=None):    #单点的scaled dot-product attention
    d_k=query.size(-1)     #获取每一个向量的维度，
    scores=torch.matmul(query,key.transpose(-2,1))/math.sqrt(d_k)   #根据上述注意力机制公式计算
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)    #将上述公式转换为sotfmax的输出
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn    #最后返回QK与V的乘积，以及QK的乘积值

class SelfAttention(nn.Module):
    def __init__(self,hid_dim,n_heads,dropout,device):
        '''
        :param hid_dim:隐藏层维度，即为上一个嵌入层的输出
        :param n_heads: 多头数目，一般为12或24
        :param dropout:dropout主要是为了过拟合而设置的,即在前向传播时,让某个激活值以一定的概率p停止工作,使模型的泛化性更强
        :param device:确定是否需要使用GPU
        '''
        super().__init__()

        self.hid_dim=hid_dim
        self.n_heads=n_heads

        assert hid_dim%n_heads==0     #设置断言，因为根据式子
        self.w_q=nn.Linear(hid_dim,hid_dim)
        self.w_k=nn.Linear(hid_dim,hid_dim)
        self.w_v=nn.Linear(hid_dim,hid_dim)

        self.fc=nn.Linear(hid_dim,hid_dim)
        self.do=nn.Dropout(dropout)

        self.scale=torch.sqrt(torch.FloatTensor([hid_dim//n_heads])).to(device)

    def forward(self,query,key,value,mask=None):
        bsz=query.shape[0]
        Q=self.w_q(query)
        K=self.w_k(key)
        V=self.w_v(value)

        Q=Q.view(bsz,-1,self.n_heads,self.hid_dim).permute(0,2,1,3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)

        energy=torch.matmul(Q,K.permute(0,1,3,2))//self.scale

        if mask is not None:
            energy=energy.masked_fill(mask=0,-1e10)

        attention=self.do(torch.softmax(energy,dim=-1))

        x=torch.matmul(attention,V)

        x=x.permute(0,2,1,3).contigous()
        x=x.view(bsz,-1,self.n_heads*(self.hid_dim//self.n_heads))
        x=self.fc(x)
        return x