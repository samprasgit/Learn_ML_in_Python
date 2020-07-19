<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1 集成学习方法](#1-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95)
- [2 深度学习中的集成方法](#2-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95)
  - [2.1 Dropout](#21-dropout)
  - [2.2 TTA](#22-tta)
  - [2.3 Snapshot](#23-snapshot)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

### 1 集成学习方法

机器学习常见的集成方法有Satcking、Bagging、Boosting,同时这些集成学习方法与具体验证集划分联系紧密.

K折交叉验证方法在CNN模型中的集成使用：

- 对预测的结果的概率值进行平均，然后解码为具体字符；     
- 对预测的字符进行投票，得到最终字符

### 2 深度学习中的集成方法

#### 2.1 Dropout

每个训练批次中，通过随机让一部分的节点停止工作,同时在预测的过程中让所有的节点都其作用,可以有效的缓解模型过拟合的情况，也可以在预测时增加模型的精度

#### 2.2 TTA 

测试集数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧，数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增，对同一个样本预测三次，然后对三次结果进行平均。     

#### 2.3 Snapshot

[Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/pdf/1704.00109v1.pdf)

在只训练了一个模型的情况下，可以使用cyclical learning rate进行训练模型，并保存精度比较好的一些checkopint，最后将多个checkpoint进行模型集成，但是此种方法可以在一定程度上提高模型精度，还需要更长的训练时间