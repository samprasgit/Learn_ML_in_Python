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



```python
'''
硬投票：对多个模型直接进行投票，不区分模型结果的相对重要度，最终投票数最多的类为最终被预测的类。
'''

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# Voting投票机制
# 硬投票：对多个模型直接进行投票，不区分模型结果的相对重要度，最终投票数最多的类为最终被预测的类
# 软投票：和硬投票原理相同，增加了设置权重的功能，可以为不同模型设置不同权重，进而区别模型不同的重要度。
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf1 = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, min_child=2,
                     subsample=0.7, colsample=0.6, objective='binary:logistic')

clf2 = RandomForestClassifier(n_estimators=50, max_depth=1, min_samples_split=4,
                              min_samples_leaf=63, oob_score=True)
clf3 = SVC(C=0.1)

# 硬投票
eclf = VotingClassifier(
    estimators=[('xgb', clf1), ('rf', clf2), ('SVC', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['XGBBoosting', 'Random Forest', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    print("Accuracy:%0.2f(+/- %0.2f) [%s] " %
          (scores.mean(), scores.std(), label))

```

输出：

```
Accuracy:0.96(+/- 0.02) [XGBBoosting] 
Accuracy:0.33(+/- 0.00) [Random Forest] 
Accuracy:0.95(+/- 0.03) [SVM] 
Accuracy:0.96(+/- 0.02) [Ensemble] 
```

```python
# 软投票：和硬投票原理相同，增加了设置权重的功能，可以为不同模型设置不同权重，进而区别模型不同的重要度。
iris = datasets.load_iris()

x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf1 = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, min_child_weight=2, subsample=0.8,
                     colsample_bytree=0.8, objective='binary:logistic')
clf2 = RandomForestClassifier(n_estimators=50, max_depth=1, min_samples_split=4,
                              min_samples_leaf=63, oob_score=True)
clf3 = SVC(C=0.1, probability=True)

# 软投票
eclf = VotingClassifier(estimators=[(
    'xgb', clf1), ('rf', clf2), ('svc', clf3)], voting='soft', weights=[2, 1, 1])
clf1.fit(x_train, y_train)

for clf, label in zip([clf1, clf2, clf3, eclf], ['XGBBoosting', 'Random Forest', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
          (scores.mean(), scores.std(), label))

```

输出：

```
Accuracy: 0.96 (+/- 0.02) [XGBBoosting]
Accuracy: 0.33 (+/- 0.00) [Random Forest]
Accuracy: 0.95 (+/- 0.03) [SVM]
Accuracy: 0.96 (+/- 0.02) [Ensemble]
```

### 5.2 分类的Srtacking/Blending融合

stacking是一种分层模型集成框架

> 以两层为例，第一层由多个基学习器组成，其输入为原始训练集，第二层的模型则是以第一层基学习器的输出作为训练集进行再训练，从而得到完整的stacking模型, stacking两层模型都使用了全部的训练数据。

```python
iris = datasets.load_iris()

'''
5-Fold Stacking

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

# 创建训练的数据集
data_0 = iris.data
data = data_0[:100, :]

target_0 = iris.target
target = target_0[:100]

# 模型中使用得到的各个模型
clfs = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(
            n_estimators=5, learning_rate=0.05, subsample=0.5, max_depth=6)
        ]

# 切割一部分数据集作为测试集
X, X_predict, y, y_predict = train_test_split(data, target,
                                              test_size=0.3, random_state=2020)

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

# 5折stacking
n_splits = 5
skf = StratifiedKFold(n_splits)
skf = skf.split(X, y)
for j, clf in enumerate(clfs):
    # 依次训练各个单模型
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(clfs)))
    for i, (train, test) in enumerate(skf):
        # 5-Fold交叉训练，使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    # 对于测试集，直接用这k个模型的预测值均值作为新的特征。
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(
        y_predict, dataset_blend_test[:, j]))

clf = LogisticRegression(solver='lbfgs')
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

print("Val acu Score of Stacking:%f" %
      (roc_auc_score(y_predict, y_submission)))
```

```
val auc Score: 1.000000
val auc Score: 0.500000
val auc Score: 0.500000
val auc Score: 0.500000
val auc Score: 0.500000
Val acu Score of Stacking:1.000000
[Finished in 1.6s]
```

Blending，其实和Stacking是一种类似的多层模型融合的形式

> 其主要思路是把原始的训练集先分成两部分，比如70%的数据作为新的训练集，剩下30%的数据作为测试集。

> 在第一层，我们在这70%的数据上训练多个模型，然后去预测那30%数据的label，同时也预测test集的label。

> 在第二层，我们就直接用这30%数据在第一层预测的结果做为新特征继续训练，然后用test集第一层预测的label做特征，用第二层训练的模型做进一步预测

其优点在于：

- 1.比stacking简单（因为不用进行k次的交叉验证来获得stacker feature）
- 2.避开了一个信息泄露问题：generlizers和stacker使用了不一样的数据集

缺点在于：

- 1.使用了很少的数据（第二阶段的blender只使用training set10%的量）
- 2.blender可能会过拟合
- 3.stacking使用多次的交叉验证会比较稳健 '''

```python
'''

Blending
[description]
'''
# 创建训练的数据集
# 创建训练的数据集
data_0 = iris.data
data = data_0[:100, :]

target_0 = iris.target
target = target_0[:100]

# 模型融合中使用到的各个单模型
clfs = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        #ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

# 切分一部分数据作为测试集
X, X_predict, y, y_predict = train_test_split(
    data, target, test_size=0.3, random_state=2020)

# 切分训练数据集为d1,d2两部分
X_d1, X_d2, y_d1, y_d2 = train_test_split(
    X, y, test_size=0.5, random_state=2020)
dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    # 依次训练各个单模型
    clf.fit(X_d1, y_d1)
    y_submission = clf.predict_proba(X_d2)[:, 1]
    dataset_d1[:, j] = y_submission
    # 对于测试集，直接用这k个模型的预测值作为新的特征。
    dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_d2[:, j]))

# 融合使用的模型
clf = GradientBoostingClassifier(
    learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_d1, y_d2)
y_submission = clf.predict_proba(dataset_d2)[:, 1]
print("Val auc Score of Blending: %f" %
      (roc_auc_score(y_predict, y_submission)))
```

```
输出：
val auc Score: 1.000000
val auc Score: 1.000000
val auc Score: 1.000000
val auc Score: 1.000000
val auc Score: 1.000000
Val auc Score of Blending: 1.000000
[Finished in 2.1s]
```

### 5.3 分类的Stacking融合 -- mlxtend

