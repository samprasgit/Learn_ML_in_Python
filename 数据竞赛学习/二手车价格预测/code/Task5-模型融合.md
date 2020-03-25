[TOC]



## 1 模型融合目标

- 对于多种调参完成的模型进行模型融合。
- 完成对于多种模型的融合，提交融合结果并打卡。

## 2 模型融合方法

模型融合是比赛后期一个重要的环节，大体来说有如下的类型方式。

1. ### 简单加权融合:

   - 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）；
   - 分类：投票（Voting)
   - 综合：排序融合(Rank averaging)，log融合

2. ### stacking/blending:

   - 构建多层模型，并利用预测结果再拟合预测。

3. ### boosting/bagging（在xgboost，Adaboost,GBDT中已经用到）:

   - 多树的提升方法

## 3 Stacking相关理论 

#### 3.1 什么是stacking

[TOC]



> stacking 是当用初始训练数据学习出若干个基学习器后，将这几个学习器的预测结果作为新的训练集，来学习一个新的学习器。

![img](https://camo.githubusercontent.com/7c71414331606f6169511b07898f44ffa7852734/687474703a2f2f6a75707465722d6f73732e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f7075626c69632f66696c65732f696d6167652f323332363534313034322f313538343434383739333233315f365479676a58776a4e622e6a7067)



将个体学习器结合在一起的时候使用的方法叫做结合策略。对于分类问题，我们可以使用投票法来选择输出最多的类。对于回归问题，我们可以将分类器输出的结果求平均值。

上面说的投票法和平均法都是很有效的结合策略，还有一种结合策略是使用另外一个机器学习算法来将个体机器学习器的结果结合在一起，这个方法就是Stacking。

在stacking方法中，我们把个体学习器叫做初级学习器，用于结合的学习器叫做次级学习器或元学习器（meta-learner），次级学习器用于训练的数据叫做次级训练集。次级训练集是在训练集上用初级学习器得到的。

#### 3.2 如何进行stacking

算法示意图：

![img](https://camo.githubusercontent.com/752adfe0742bd1b3a83d4126aba9cefedebaf9be/687474703a2f2f6a75707465722d6f73732e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f7075626c69632f66696c65732f696d6167652f323332363534313034322f313538343434383830363738395f31456c527448616163772e6a7067)

> 引用自 西瓜书

- 过程1-3 是训练出来个体学习器，也就是初级学习器。
- 过程5-9是 使用训练出来的个体学习器来得预测的结果，这个预测的结果当做次级学习器的训练集。
- 过程11 是用初级学习器预测的结果训练出次级学习器，得到我们最后训练的模型。

### 3.3 Stacking训练过程/方法







K-折交叉验证： 训练：

![img](https://camo.githubusercontent.com/5ff9d18b7192204faec52a4a89a11efea4e1d914/687474703a2f2f6a75707465722d6f73732e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f7075626c69632f66696c65732f696d6167652f323332363534313034322f313538343434383831393633325f59764a4f584d6b3032502e6a7067)

预测：

![img](https://camo.githubusercontent.com/cc0d751cdf81ad08c9cea53bbd31c7f036d1db77/687474703a2f2f6a75707465722d6f73732e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f7075626c69632f66696c65732f696d6167652f323332363534313034322f313538343434383832363230335f6b384b5079396e3744392e6a7067)



## 4 回归/概率融合

## 5 分类模型融合

```
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
```



### 5.1 Voting投票机制

Voting即投票机制，分为软投票和硬投票两种，其原理采用少数服从多数的思想：

- ```
  硬投票：对多个模型直接进行投票，不区分模型结果的相对重要度，最终投票数最多的类为最终被预测的类
  ```

- ```
  软投票：和硬投票原理相同，增加了设置权重的功能，可以为不同模型设置不同权重，进而区别模型不同的重要度。
  ```



