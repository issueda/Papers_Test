# -*- coding: utf-8 -*-

import codecs
import sys

import gensim
from tqdm import tqdm


class Sentences(object):  # 文件读取
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):  # 一行一行读出
        for line in tqdm(codecs.open(self.filename, "r", "utf-8")):
            yield line.strip().split()




def main(path):
    sentences = Sentences(path) # 获取
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=7, sg=1,  # word2vec 训练
                                   negative=5, iter=1, max_vocab_size=20000)
    model.save(path + ".w2v")   # 保存
    # model.wv.save_word2vec_format("word_vectors/" + domain + ".txt", binary=False)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "/home/uyplayer/Github/ABAE/abae-pytorch/data/Electronics_5.json.txt"


    print("Training w2v on dataset", path)

    main(path)

    print("Training done.")

    model = gensim.models.Word2Vec.load("/home/uyplayer/Github/ABAE/abae-pytorch/word_vectors/Electronics_5.json.txt.w2v")  # 加载模型

    for word in ["he", "love", "looks", "buy", "laptop"]:
        if word in model.wv.vocab:
            print(word, [w for w, c in model.wv.similar_by_word(word=word)])
        else:
            print(word, "not in vocab")
