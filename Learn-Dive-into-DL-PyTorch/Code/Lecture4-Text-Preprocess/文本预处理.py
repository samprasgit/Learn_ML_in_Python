# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai
# @Editor  : sublime 3
#
"""

文本预处理

文本是一类序列数据，一篇文章可以看作是字符或单词的序列，本节将介绍文本数据的常见预处理步骤，预处理通常包括四个步骤：

1. 读入文本
2. 分词
3. 建立字典，将每个词映射到一个唯一的索引（index）
4. 将文本从词的序列转换为索引的序列，方便输入模型

"""

import re
import collections
import sys


# 读取数据集

path = "../../dataset/jaychou_lyrics.txt"


def read_time_machine():
    with open(path, 'r', encoding="utf-8") as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines
# strip去除首尾指定元素，默认\n和空格
# ltrip去除首元素，rstrip去除末尾元素

lines = read_time_machine()
print('# sentences %d' % len(lines))

# 分词
# 对每个句子进行分词，也就是将一个句子划分成若干个词（token），转换为一个词的序列


def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type ' + token)

tokens = tokenize(lines)
tokens[0:2]

# 建立词典
# 为了方便模型处理，我们需要将字符串转换为数字。因此我们需要先构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号。
#


class Vocab(object):

    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                              if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

# 统计词频
def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数


# 查看构架的字典效果
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])

# 将词转为索引

# 使用字典，我们可以将原文本中的句子从单词序列转换为索引序列
for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

# 我们前面介绍的分词方式非常简单，它至少有以下几个缺点:
#
# 1. 标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
# 2. 类似“shouldn't", "doesn't"这样的词会被错误地处理
# 3. 类似"Mr.", "Dr."这样的词会被错误地处理
# 我们可以通过引入更复杂的规则来解决这些问题，但是事实上，有一些现有的工具可以很好地进行分词，我们在这里简单介绍其中的两个：spaCy和NLTK。

text = "Mr. Chen doesn't agree with my suggestion."


#  spacy
#  nltk
