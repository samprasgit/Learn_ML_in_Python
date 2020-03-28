<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [加权融合](#%E5%8A%A0%E6%9D%83%E8%9E%8D%E5%90%88)
- [Stacking融合](#stacking%E8%9E%8D%E5%90%88)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

```python
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
%matplotlib inline

import itertools
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
```

数据读取

```python
train_data = pd.read_csv("../data/train.csv", sep=' ')
test_data = pd.read_csv("../data/testA.csv", sep=' ')
print(Train_data.shape)
print(TestA_data.shape)
```

```
(150000, 31)
(50000, 30)
```

数字特征选取

```python
umerical_cols = train_data.select_dtypes(exclude='object').columns
numerical_cols
```

```
Index(['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType',
       'gearbox', 'power', 'kilometer', 'regionCode', 'seller', 'offerType',
       'creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
       'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14'],
      dtype='object')
```

```python
feature_cols = [col for col in numerical_cols if col not in [
    'SaleID', 'name', 'regDate', 'price']]
```

```python
X_data = train_data[feature_cols]
Y_data = train_data['price']
X_test = test_data[feature_cols]
print(X_data.shape, X_test.shape)
```

label信息统计

```python
def Sta_inf(data):
    print('_min', np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))
print("label的统计信息：")
Sta_inf(Y_data)
```

缺失值处理

Bodytype fueltype offerType 几个特征为类别特征

```python
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)
```

建立模型

```python
def build_model_lin(x_train, y_train):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x_train, y_train)
    return reg_model


def build_model_ridge(x_train, y_train):
    reg_model = linear_model.Ridge(alpha=0.8)  # alphas=range(1,100,5)
    reg_model.fit(x_train, y_train)
    return reg_model


def build_model_lasso(x_train, y_train):
    reg_model = linear_model.LassoCV()
    reg_model.fit(x_train, y_train)
    return reg_model


def build_model_gbdt(x_train, y_train):
    estimator = GradientBoostingRegressor(
        loss='ls', subsample=0.85, max_depth=5, n_estimators=100)
    param_grid = {
        'learning_rate': [0.05, 0.08, 0.1, 0.2],
    }
    gbdt = GridSearchCV(estimator, param_grid, cv=3)
    gbdt.fit(x_train, y_train)
    print(gbdt.best_params_)
    # print(gbdt.best_estimator_ )
    return gbdt


def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=120, learning_rate=0.08, gamma=0, subsample=0.8,
                             colsample_bytree=0.9, max_depth=5)  # , objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model


def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=63, n_estimators=100)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm
```

XGBoost的五折交叉回归验证实现 

```python
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.9, max_depth=7)
scores_train = []
scores = []

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

for train_ind, val_ind in skf.split(X_data, Y_data):

    train_x = X_data.iloc[train_ind].values
    train_y = Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pre_train_xgb = xgr.predict(train_x)
    pre_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y, pre_train_xgb)
    scores_train.append(score_train)
    scores = mean_absolute_error(val_y, pre_xgb)


print("Train MAE:", np.mean(scores_train))
print("Val MAE:", np.mean(scores))
```

```python
Train MAE: 598.499217582531
Val MAE: 748.8762653225563
```

划分数据集，并用多种方法训练和预测

```python
# 划分数据及，并运用多种方法训练和预测
x_train, x_val, y_train, y_val = train_test_split(
    X_data, Y_data, test_size=0.3)

# train and predict
print("Predict linea...")
model_lin = build_model_lin(x_train, y_train)
val_lin = model_lin.predict(x_val)
sub_lin = model_lin.predict(X_test)

print('Predict Ridge...')
model_ridge = build_model_ridge(x_train, y_train)
val_ridge = model_ridge.predict(x_val)
sub_ridge = model_ridge.predict(X_test)

print("Predict Lasso...")
model_lasso = build_model_lasso(x_train, y_train)
val_lasso = model_lasso.predict(x_val)
sub_lasso = model_lasso.predict(X_test)

print("Predict GBDT...")
model_gbdt = build_model_gbdt(x_train, y_train)
val_gbdt = model_gbdt.predict(x_val)
sub_gbdt = model_gbdt.predict(X_test)
```

```python
# 一般比赛中效果最显著的两种方法
print("Predict XGB...")
model_xgb = build_model_xgb(x_train, y_train)
val_xgb = model_xgb.predict(x_val)
sub_xgb = model_xgb.predict(X_test)

print("Predict lgb'")
model_lgb = build_model_lgb(x_train, y_train)
val_lgb = model_lgb.predict(x_val)
sub_lgb = mdoel_lgb.predict(X_test)
```

### 加权融合

```python
def Weighted_method(test_pre1, testZ_pre2, test_pred_3, w=[1 / 3, 1 / 3, 1 / 3]):
    Weighted_result = w[
        0] * pd.Series(test_pre1) + w[1] * pd.Series(test_pre2) + w[2] * pd.Series(test_pre2)
    return Weighted_result

# init the weight
w = [0.3, 0.4, 0.3]

# 测试验证集的准确度
val_pre = Weighted_method(val_lgb, val_xgb, val_gbdt, w)
MAE_Weighted = mean_absolute_error(y_val, val_pre)
print("MAE of Weighted of val:", MAE_Weighted)
```

```
MAE of Weighted of val: 720.004436282035
Sta inf:
_min -100.28725676215899
_max: 88204.53903020486
_mean 5932.854726866527
_ptp 88304.82628696701
_std 7364.744887330821
_var 54239467.25546548
```

```python
# 与简单的LR（线性回归）进行对比
val_lr_pred = model_lr.predict(x_val)
MAE_lr = mean_absolute_error(y_val, val_lr_pred)
print('MAE of lr:', MAE_lr)
```

### Stacking融合

```python 
# 第一层

train_lgb_pred = model_lgb.predict(x_train)
train_xgb_pred = model_xgb.predict(x_train)
train_gbdt_pred = model_gbdt.predict(x_train)

Stack_X_train = pd.DataFrame()
Stack_X_train['Method_1'] = train_lgb_pred
Stack_X_train['Method_2'] = train_xgb_pred
Stack_X_train['Method_3'] = train_gbdt_pred

Stack_X_val = pd.DataFrame()
Stack_X_val['Method_1'] = val_lgb
Stack_X_val['Method_2'] = val_xgb
Stack_X_val['Method_3'] = val_gbdt

Stack_X_test = pd.DataFrame()
Stack_X_test['Method_1'] = sub_lgb
Stack_X_test['Method_2'] = sub_xgb
Stack_X_test['Method_3'] = sub_gbdt

Stack_X_test.head()
```

|      |   Method_1   |   Method_2   | **Method_3** |
| :--: | :----------: | :----------: | :----------: |
|  0   | 39690.496183 | 38247.667969 | 39684.617194 |
|  1   |  275.013090  |  251.308945  |  197.147617  |
|  2   | 7395.503934  | 7373.573730  | 7167.100073  |
|  3   | 11541.447705 | 11416.806641 | 12110.959602 |
|  4   |  576.633811  |  541.591614  |  463.119185  |

```python
# level2-method
model_lin_Stacking = build_model_lin(Stack_X_train, y_train)

# 训练集
train_pre_Stacking = model_lin_Stacking.predict(Stack_X_train)
print('MAE of Stackin-Lin:', mean_absolute_error(y_train, train_pre_Stacking))

# 验证集
val_pre_Stacking = model_lin_Stacking.predict(Stack_X_val)
print("MAE of Satcking_Lin: ", mean_absolute_error(y_val, val_pre_Stacking))

# 测试集
print("Predicting Stacking_Lin...")
sub_Stacking = model_lin_Stacking.predict(Stack_X_test)
```

```
MAE of Stackin-Lin: 629.3776240456978
MAE of Satcking_Lin:  715.0568885660866
Predicting Stacking_Lin...
```

```python
# 祛除小的预测值
sub_Stacking[sub_Stacking < 10] = 10

sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = sub_Stacking
sub.to_csv('sub_Satcking.csv', index=False)

print('Sta_inf:')
Sta_inf(sub_Stacking)
```

```
Sta_inf:
_min 10.0
_max: 92368.25512155247
_mean 5932.778213107892
_ptp 92358.25512155247
_std 7427.800544016455
_var 55172220.92169115
```

经验总结

比赛的融合这个问题，其实涉及多个层面，也是提分和提升模型鲁棒性的一种重要方法：

- 1）**结果层面的融合**，这种是最常见的融合方法，其可行的融合方法也有很多，比如根据结果的得分进行加权融合，还可以做Log，exp处理等。在做结果融合的时候，有一个很重要的条件是模型结果的得分要比较近似，然后结果的差异要比较大，这样的结果融合往往有比较好的效果提升。
- 2）**特征层面的融合**，这个层面其实感觉不叫融合，准确说可以叫分割，很多时候如果我们用同种模型训练，可以把特征进行切分给不同的模型，然后在后面进行模型或者结果融合有时也能产生比较好的效果。
- 3）**模型层面的融合**，模型层面的融合可能就涉及模型的堆叠和设计，比如加Staking层，部分模型的结果作为特征输入等，这些就需要多实验和思考了，基于模型层面的融合最好不同模型类型要有一定的差异，用同种模型不同的参数的收益一般是比较小的。

