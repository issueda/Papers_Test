# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2019/11/19 下午8:11
# Tool ：PyCharm


from reader import get_centroids, get_w2v, read_data_tensors
import torch
from torch.nn.parameter import Parameter

# w2v_model = get_w2v("/home/uyplayer/Github/ABAE/abae-pytorch/word_vectors/Electronics_5.json.txt.w2v")  # 加载word2vec模型
# wv_dim = w2v_model.vector_size  # 获取长度
# y = torch.zeros(wv_dim, 1)
p = Parameter(torch.Tensor(200, 200)) # Parameters 是 Variable 的子类
print(p.data)
print(p)