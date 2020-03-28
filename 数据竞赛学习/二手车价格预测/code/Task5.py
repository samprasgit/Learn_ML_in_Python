# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

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
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, SparsePCA

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

# 数据读取
train_data = pd.read_csv("../data/train.csv", sep=' ')
test_data = pd.read_csv("../data/testA.csv", sep=' ')

print(train_data.shape)
print(test_data.shape)

print(train_data.shape)

numerical_cols = train_data.select_dtypes(exclude='object').columns
numerical_cols

feature_cols = [col for col in numerical_cols if col not in [
    'SaleID', 'name', 'regDate', 'price']]

X_data = train_data[feature_cols]
Y_data = train_data['price']
X_test = test_data[feature_cols]
print(X_data.shape, X_test.shape)


def Sta_inf(data):
    print('_min', np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))
print("label的统计信息：")
Sta_inf(Y_data)

X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)


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

# XGBoost的五折交叉回归验证实现
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


# 一般比赛中效果最显著的两种方法
print("Predict XGB...")
model_xgb = build_model_xgb(x_train, y_train)
val_xgb = model_xgb.predict(x_val)
sub_xgb = model_xgb.predict(X_test)

print("Predict lgb'")
model_lgb = build_model_lgb(x_train, y_train)
val_lgb = model_lgb.predict(x_val)
sub_lgb = mdoel_lgb.predict(X_test)

# 加权融合


def Weighted_method(test_pre1, test_pre2, test_pre3, w=[1 / 3, 1 / 3, 1 / 3]):
    Weighted_result = w[
        0] * pd.Series(test_pre1) + w[1] * pd.Series(test_pre2) + w[2] * pd.Series(test_pre3)
    return Weighted_result

# Init the Weight
w = [0.3, 0.4, 0.3]

# 测试验证集准确度
val_pre = Weighted_method(val_lgb, val_xgb, val_gbdt, w)
MAE_Weighted = mean_absolute_error(y_val, val_pre)
print('MAE of Weighted of val:', MAE_Weighted)

# 预测数据部分
subA = Weighted_method(subA_lgb, subA_xgb, subA_gbdt, w)
print('Sta inf:')
Sta_inf(subA)
# 生成提交文件
sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = subA
sub.to_csv('sub_Weighted.csv', index=False)


# 与简单的LR（线性回归）进行对比
val_lr_pred = model_lr.predict(x_val)
MAE_lr = mean_absolute_error(y_val, val_lr_pred)
print('MAE of lr:', MAE_lr)


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


# 祛除小的预测值
sub_Stacking[sub_Stacking < 10] = 10

sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = sub_Stacking
sub.to_csv('sub_Satcking.csv', index=False)

print('Sta_inf:')
Sta_inf(sub_Stacking)
