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

