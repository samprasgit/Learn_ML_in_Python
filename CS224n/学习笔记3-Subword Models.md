<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [学习任务](#%E5%AD%A6%E4%B9%A0%E4%BB%BB%E5%8A%A1)
- [语言模型](#%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)
- [n-gram语言模型](#n-gram%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)
- [subword embedding](#subword-embedding)
- [GloVe Model](#glove-model)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

### 学习任务

- 回归Wordvec模型
- 介绍 count based global matrix factorization 方法
- 介绍 GloVe 模型



### 语言模型

对于一个文本中出现的单词 𝑤𝑖的概率，他更多的依靠的是前 𝑛 个单词，而不是这句话中前面所有的单词
$$
P\left(w_{1}, \ldots, w_{m}\right)=\prod_{i=1}^{i=m} P\left(w_{i} \mid w_{1}, \ldots, w_{i-1}\right) \approx \prod_{i=1}^{i=m} P\left(w_{i} \mid w_{i-n}, \ldots, w_{i-1}\right)
$$


在翻译系统中就是对于输入的短语，通过对所有的输出的语句进行评分，得到概率最大的那个输出，作为预测的概率

### n-gram语言模型

在N-Gram语言模型中， count 可以用来表示单词出现的频率，这个模型与条件概率密切相关，其中，
$$
\begin{aligned}
p\left(w_{2} \mid w_{1}\right) &=\frac{\operatorname{count}\left(w_{1}, w_{2}\right)}{\operatorname{count}\left(w_{1}\right)} \\
p\left(w_{3} \mid w_{1}, w_{2}\right) &=\frac{\operatorname{count}\left(w_{1}, w_{2}, w_{3}\right)}{\operatorname{count}\left(w_{1}, w_{2}\right)}
\end{aligned}
$$
将连续单词出现的频率作为概率，然后通过条件概率的形式就可以预测出下一个单词。

比较难得是选取前面多少个单词。



### subword embedding

论文 `《Enriching Word Vectors with Subword Information》` 中，作者提出通过增加字符级信息来训练词向量

下图给出了该方法在维基百科上训练的词向量在相似度计算任务上的表现（由人工评估模型召回的结果）。`sisg-` 和 `sisg` 模型均采用了 `subword embedding`，区别是：对于未登录词，`sisg-` 采用零向量来填充，而 `sisg` 采用 `character n-gram embedding` 来填充。

![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/word_representation/word2vec_char.png)

单词拆分：每个单词表示为一组 `character n-gram` 字符（不考虑顺序），以单词 `where`、 `n=3` 为例：

- 首先增加特殊的边界字符 `<` （单词的左边界）和 `>` （单词的右边界）。
- 然后拆分出一组 `character n-gram` 字符：`<wh, whe,her,ere,re>` 。
- 最后增加单词本身：`<where>`。

为了尽可能得到多样性的 `character n-gram` 字符，作者抽取了所有 `3<= n <= 6` 的 `character n-gram` 。以单词 `mistake` 为例：

```
<mi,mis,ist,sta,tak,ake,ke>,   // n = 3
<mis,mist,ista,stak,take,ake>, // n = 4
<mist,mista,istak,stake,take>, // n = 5
<mista,mistak,istake,stake>,   // n = 6
<mistake>                      // 单词本身
```

注意：这里的 `take` 和 `<take>` 不同。前者是某个`character n-gram`，后者是一个单词。

一旦拆分出单词，则：

- 词典 扩充为包含所有单词和 `N-gram` 字符。
- 网络输入包含单词本身以及该单词的所有 `character n-gram` ，网络输出仍然保持为单词本身。

模型采用 `word2vec` ，训练得到每个`character n-gram embedding` 。最终单词的词向量是其所有 `character n-gram embedding`包括其本身 `embedding` 的和（或者均值）。

如：单词 `where` 的词向量来自于下面`embedding` 之和：

- 单词 `<where>` 本身的词向量。
- 一组 `character n-gram` 字符 `<wh, whe,her,ere,re>` 的词向量。

利用字符级信息训练词向量有两个优势：

- 有利于低频词的训练。

  低频词因为词频较低，所以训练不充分。但是低频词包含的 `character n-gram` 可能包含某些特殊含义并且得到了充分的训练，因此有助于提升低频词的词向量的表达能力。

- 有利于获取 `OOV` 单词（未登录词：不在词汇表中的单词）的词向量。

  对于不在词汇表中的单词，可以利用其 `character n-gram` 的`embedding` 来获取词向量。

### GloVe Model



