# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2019/10/22 下午4:02
# Tool ：PyCharm

# python script/prep_oov.py

import argparse
import numpy as np
import json
import os
import subprocess

def fill_np_embedding(embed_file,word_idx_fn,out_np_file):
    # 加载 word_index
    with open(word_idx_fn) as word_ind:
        word_indx = json.load(word_ind)
    embedding = np.load(embed_file) # 加载 embedding

    with open(out_np_file) as out_npFile:
        for line in out_npFile:
            line_split = line.rstrip().split(' ')
            if len(line_split) == 2: # 没有向量就continue
                continue
            if line_split[0] in word_indx:
                embedding[word_indx[line_split[0]]] = np.array([float(r) for r in line_split[1:] ])   # 修改位置

    np.save(embed_file,embedding.astype("float32"))

# 获取根目录
proj_root = os.path.abspath('..')+"/DE-CNN"
# 开始设置终端参数
parser = argparse.ArgumentParser()
parser.add_argument('--laptop_emb_np', type=str, default="laptop_emb.vec.npy")
parser.add_argument('--restaurant_emb_np', type=str, default="restaurant_emb.vec.npy")
parser.add_argument('--out_dir', type=str, default=proj_root+"/data/prep_data/")
parser.add_argument('--laptop_oov', type=str, default="laptop_oov.vec")
parser.add_argument('--restaurant_oov', type=str, default="restaurant_oov.vec")
parser.add_argument('--word_idx', type=str, default="word_idx.json")
args = parser.parse_args()

fill_np_embedding(args.out_dir+args.laptop_emb_np, args.out_dir+args.word_idx, args.out_dir+args.laptop_oov)

fill_np_embedding(args.out_dir+args.restaurant_emb_np, args.out_dir+args.word_idx, args.out_dir+args.restaurant_oov)
