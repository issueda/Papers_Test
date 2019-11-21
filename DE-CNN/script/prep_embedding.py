# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2019/10/22 上午11:35
# Tool ：PyCharm

# Run prep_embedding.py to build numpy files for general embeddings and domain embeddings

import argparse
import numpy as np
import json
import os
import sys
import subprocess

# 形成numpy向量文件
def gen_numpy_embedding(file_embendding,word_idx_file,out_path,dim=300):

    # 加载 word_index
    with open(word_idx_file) as file_word_id:
        words_indix = json.load(file_word_id) # 加载json文件
    embedding = np.zeros((len(words_indix)+2,dim))

    # 加载预训练好的的embedding
    with open(file_embendding) as file_em:
        for line in file_em:
            line_split = line.rstrip().split(' ')
            if len(line_split)==2: # 表示该行没有向量
                continue
            if line_split[0] in words_indix: # 如果单词在word_index里面
                embedding[words_indix[line_split[0]]] = np.array([float(r) for r in line_split[1:]])   # 形成我们的embedding

    with open(out_path + ".oov.txt", "w") as save_em:
        for w in words_indix: # 提出不再words_indix里面的单词并写入单词
            if embedding[words_indix[w]].sum()==0.:
                save_em.write(w + "\n")
    np.save(out_path+".npy",embedding.astype("float32")) # 保存我们形成的响亮文件


# 获取根目录
proj_root = os.path.abspath('..')
# 开始设置终端参数
parser = argparse.ArgumentParser()
parser.add_argument("--emb_pre_dir",type=str,default=proj_root+"/data/embedding/")
parser.add_argument("--out_embed_dir",type=str,default=proj_root+"/data/prep_data/")
parser.add_argument("--gen_pre_embed",type=str,default="gen.vec")
parser.add_argument("--laptop_pre_embed",type=str,default="laptop_emb.vec")
parser.add_argument("--restaurant_pre_emb",type=str,default="restaurant_emb.vec")
parser.add_argument("--word_idx",type=str,default="word_idx.json")
parser.add_argument("--gen_dim",type=int,default=300)
parser.add_argument("--domain_dim",type=int,default=100)
args = parser.parse_args() # 初始化结束


# 开始执行函数并形成我们的embedding

gen_numpy_embedding(args.emb_pre_dir+args.gen_pre_embed, args.out_embed_dir+args.word_idx, args.out_embed_dir+args.gen_pre_embed, args.gen_dim)

gen_numpy_embedding(args.emb_pre_dir+args.laptop_pre_embed, args.out_embed_dir+args.word_idx, args.out_embed_dir+args.laptop_pre_embed, args.domain_dim)

gen_numpy_embedding(args.emb_pre_dir+args.restaurant_pre_emb, args.out_embed_dir+args.word_idx, args.out_embed_dir+args.restaurant_pre_emb, args.domain_dim)























