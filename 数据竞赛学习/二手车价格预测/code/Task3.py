# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
# 读取数据
train = pd.read_csv('../data/train.csv', sep=' ')
test = pd.read_csv('../data/testA.csv', sep=' ')
print(train.shape)
print(test.shape)


# 删除异常值
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n

# 我们可以删掉一些异常数据，以 power 为例。
# 这里删不删同学可以自行判断

train = outliers_proc(train, 'power', scale=3)
# 特征构造
# 训练集和测试集放在一起，方便构造特征
train['train'] = 1
test['train'] = 0
data = pd.concat([train, test], ignore_index=True, sort=False)

# 使用时间：data['creatDate'] - data['regDate']，反应汽车使用时间，一般来说价格与使用时间成反比
# 不过要注意，数据里有时间出错的格式，所以我们需要 errors='coerce'
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') -
                     pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days

# 计算特征缺失比例
(data.isnull().sum() / len(data)).sort_values()
# 从邮编中提取城市信息，因为是德国的数据，所以参考德国的邮编，相当于加入了先验知识
data['city'] = data['regionCode'].apply(lambda x: str(x)[:-3])


# 计算某品牌的销售统计量
# 这里要以 train 的数据计算统计量
train_gb = train.groupby("brand")
all_info() = {}
for kind, kind_data in train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data['price'].max()
    info['brand_price_mediani'] = kind_data['price'].median()
    info['brand_price_min'] = kind_data['price'].min()
    info['brand_price_sum'] = kind_data['price'].sum()
    info['brand_price_stddev'] = kind_data['price'].std()
    info['brand_price_average'] = round(
        kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info

brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"indesx": "brand"})
data = data.merge(brand_fe, how='left', on='brand')


bin = [i * 10 for i in range(31)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data[['power_bin', 'power']].head()
# 目前的数据其实已经可以给树模型使用了，所以我们导出一下
data.to_csv('data_for_tree.csv', index=0)

data['power'].plot.hist()

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data['power'] = np.log(data[
    'power'] + 1)
data['power'] = ((data['power'] - np.min(data['power'])) /
                 (np.max(data['power']) - np.min(data['power'])))
data.power.plot.hist()

data['kilometer'] = ((data['kilometer'] - np.min(data['kilometer'])) /
                     (np.max(data['kilometer']) - np.min(data['kilometer'])))
data['kilometer'].plot.hist()


def max_min(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

data['brand_amount'] = ((data['brand_amount'] - np.min(data['brand_amount'])) /
                        (np.max(data['brand_amount']) - np.min(data['brand_amount'])))
data['brand_price_average'] = ((data['brand_price_average'] - np.min(data['brand_price_average'])) /
                               (np.max(data['brand_price_average']) - np.min(data['brand_price_average'])))
data['brand_price_max'] = ((data['brand_price_max'] - np.min(data['brand_price_max'])) /
                           (np.max(data['brand_price_max']) - np.min(data['brand_price_max'])))
data['brand_price_median'] = ((data['brand_price_median'] - np.min(data['brand_price_median'])) /
                              (np.max(data['brand_price_median']) - np.min(data['brand_price_median'])))
data['brand_price_min'] = ((data['brand_price_min'] - np.min(data['brand_price_min'])) /
                           (np.max(data['brand_price_min']) - np.min(data['brand_price_min'])))
data['brand_price_std'] = ((data['brand_price_std'] - np.min(data['brand_price_std'])) /
                           (np.max(data['brand_price_std']) - np.min(data['brand_price_std'])))
data['brand_price_sum'] = ((data['brand_price_sum'] - np.min(data['brand_price_sum'])) /
                           (np.max(data['brand_price_sum']) - np.min(data['brand_price_sum'])))

# 对类别特征进行 OneEncoder
data = pd.get_dummies(data, columns=[
                      'model', 'brand_price', 'brand', 'fuelType', 'gearbox', 'notRepairedDamage', 'power_bin'])
# 这份数据可以给 LR 用
data.to_csv('data_for_lr.csv', index=0)
