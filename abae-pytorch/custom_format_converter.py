# -*- coding: utf-8 -*-
import json
from functools import lru_cache

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm

tokenizer = TweetTokenizer(preserve_case=False)
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))


@lru_cache(1000000000)  # 内存设置
def lemmatize(w):
    return lemmatizer.lemmatize(w) # 单词变体还原


def read_amazon_format(path, sentence=True):   # 数据转换函数
    with open(path + ("" if sentence else "-full_text") + ".txt", "w+") as wf:  # 打开我们写入对象

        for line in tqdm(open(path)):  # 开始读出我们原数据
            text = json.loads(line.strip())["reviewText"].replace("\n", " ") # 获取一行数据，并去掉换行付
            sentences = sent_tokenize(text) # 分句子
            tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences] # 一条数据开始分词
            lemmatized_sentences = [[lemmatize(word) for word in s if not word in stops and str.isalpha(word)] #  形成多位的数组
                                    for s in tokenized_sentences]

            for sentence in lemmatized_sentences: # 开始写入到我们文件 一行一行的写的
                wf.write(" ".join(sentence) + "\n" if sentence else " ")

            if not sentence:
                wf.write("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "/home/uyplayer/Github/ABAE/abae-pytorch/data/Electronics_5.json"

    read_amazon_format(path, sentence=True)
