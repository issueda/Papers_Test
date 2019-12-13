# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2019/12/13 上午9:59
# Tool ：PyCharm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __int__(self,embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function="dot_product", dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention,self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function

        self.w_k = nn.Linear(embed_dim,n_head*hidden_dim)
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if score_function == "mlp":
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif score_function == "bi_linear":
            self.weight = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim))
        else:
            self.register_parameter('weight',None)
        self.register_parameter()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.unoform_(-stdv,stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q,1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]

        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_k(q).view(mb_size,q_len,self.n_head,self.hidden_dim)
        qx = qx.permute(2,0,1,3).contiguous().view(-1,q_len,self.hidden_dim)

        if self.score_function == 'dot_product':
            kt = kx.permute(0,2,1)
            score = torch.bmm(qx,kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx,kt)
            score = torch.div(qkt,math.sqrt(self.hidden_dim))






























