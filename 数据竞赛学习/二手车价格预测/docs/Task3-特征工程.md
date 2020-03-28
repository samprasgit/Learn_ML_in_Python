<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [特征工程内容](#%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E5%86%85%E5%AE%B9)
- [1.数据理解](#1%E6%95%B0%E6%8D%AE%E7%90%86%E8%A7%A3)
  - [1.1.定性数据：描述性质](#11%E5%AE%9A%E6%80%A7%E6%95%B0%E6%8D%AE%E6%8F%8F%E8%BF%B0%E6%80%A7%E8%B4%A8)
- [2.数据清洗](#2%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97)
  - [2.1.特征变换](#21%E7%89%B9%E5%BE%81%E5%8F%98%E6%8D%A2)
  - [2.2异常值处理：减少脏数据](#22%E5%BC%82%E5%B8%B8%E5%80%BC%E5%A4%84%E7%90%86%E5%87%8F%E5%B0%91%E8%84%8F%E6%95%B0%E6%8D%AE)
  - [2.3缺失值处理](#23%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86)
    - [2.4其他](#24%E5%85%B6%E4%BB%96)
- [3.特征构造](#3%E7%89%B9%E5%BE%81%E6%9E%84%E9%80%A0)
  - [3.1构造统计量特征](#31%E6%9E%84%E9%80%A0%E7%BB%9F%E8%AE%A1%E9%87%8F%E7%89%B9%E5%BE%81)
  - [3.2时间特征](#32%E6%97%B6%E9%97%B4%E7%89%B9%E5%BE%81)
  - [3.3地理信息](#33%E5%9C%B0%E7%90%86%E4%BF%A1%E6%81%AF)
  - [3.4非线性变换](#34%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2)
  - [3.5特征组合](#35%E7%89%B9%E5%BE%81%E7%BB%84%E5%90%88)
  - [3.6数据分箱](#36%E6%95%B0%E6%8D%AE%E5%88%86%E7%AE%B1)
- [4.特征选择](#4%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9)
  - [4.1过滤式](#41%E8%BF%87%E6%BB%A4%E5%BC%8F)
  - [4.2包裹式](#42%E5%8C%85%E8%A3%B9%E5%BC%8F)
  - [4.3嵌入式](#43%E5%B5%8C%E5%85%A5%E5%BC%8F)
- [5.类别不平衡](#5%E7%B1%BB%E5%88%AB%E4%B8%8D%E5%B9%B3%E8%A1%A1)
- [6.降维](#6%E9%99%8D%E7%BB%B4)
- [代码实战](#%E4%BB%A3%E7%A0%81%E5%AE%9E%E6%88%98)
  - [特征筛选](#%E7%89%B9%E5%BE%81%E7%AD%9B%E9%80%89)
    - [特征重要性排名-- Yellowbrick特征可视化库](#%E7%89%B9%E5%BE%81%E9%87%8D%E8%A6%81%E6%80%A7%E6%8E%92%E5%90%8D---yellowbrick%E7%89%B9%E5%BE%81%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BA%93)
    - [自动特征筛选工具](#%E8%87%AA%E5%8A%A8%E7%89%B9%E5%BE%81%E7%AD%9B%E9%80%89%E5%B7%A5%E5%85%B7)
    - [过滤式](#%E8%BF%87%E6%BB%A4%E5%BC%8F)
    - [包裹式](#%E5%8C%85%E8%A3%B9%E5%BC%8F)
    - [嵌入式](#%E5%B5%8C%E5%85%A5%E5%BC%8F)
- [7.经验总结](#7%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



> **特征工程(Feature Engineering):将数据转换为能更好地表示潜在问题 的特征，从而提高机器学习性能。**

## 特征工程内容

## 1.数据理解

> 探索数据，了解数据，主要在EDA阶段完成

### 	1.1.定性数据：描述性质

- 按名称分类

- 有序分类

  ### 1.2.定量数据：描述数量

## 2.数据清洗

> 提高数据质量，降低算法用错数据建模的风险

### 2.1.特征变换

- 定性变量编码

  Label Encoder;

  Onehot Endcoder;

  Distribution Coding

- 标准化和归一化

  Z分数标准化

  min-max归一化
  
  针对幂律分布，可以采用公式$log(\frac{1+x}{1+median})$

### 2.2异常值处理：减少脏数据

- 简单数据

  describe()的统计描述

  散点图

- 3$\sigma$原则、箱线图截断

- 利用模型进行离群点检测

  聚类

  K近邻

  OneClass SVM

  Isolation Forest

### 2.3缺失值处理

a)不处理：少量样本缺失、XGBoost等树模型

b)删除：大量样本缺失

c)补全：

​	均值、中位数、众数补全

​	高维映射（one hot）

​	模型预测-RF

​	最近邻补全

​	矩阵补全（R-SVD）

#### 2.4其他	

​	删除无效列

​	更改dtyeps

​	删除列中的字符串

​	将时间戳从字符串转换为日期时间格式等	

## 3.特征构造

> 增强数据表达，增加先验知识

### 3.1构造统计量特征

​	计数、求和、比例、标准差

### 3.2时间特征

​	绝对时间	

​	相对时间

​	节假日

​	双休日

### 3.3地理信息

​	分箱

​	分布编码

### 3.4非线性变换

​	log

​	平方

​	根号

### 3.5特征组合

​	特征交叉

### 3.6数据分箱

​	等频/等距分桶

​	Best-KS 分桶

​	卡方分桶

## 4.特征选择

> 降低噪声，平滑预测能力和计算复杂度，增强模型预测性能

### 4.1过滤式

先用特征选择方法对初识特征进行过滤然后再训练学习器，特征选择过程与后续学习器无关。

Relief/方差选择/相关系数/卡方检验/互信息法

### 4.2包裹式

直接把最终将要使用的学习器的性能作为衡量特征子集的评价准则，其目的在于为给定学习器选择最有利于其性能的特征子集

Las Vegas Wrapper(LVM)

### 4.3嵌入式

结合过滤式和包裹式方法，将特征选择与学习器训练过程融为一体，两者在同一优化过程中完成，即学习器训练过程中自动进行了特征选择。

LR+L1或决策树

### 4.4自动特征筛选工具



## 5.类别不平衡

1. ### 扩充数据集;

2. ### 尝试其他评价指标:AUC等; 

3. ### 调整$\theta$值;

4. ### 重采样:过采样/欠采样;

5. ### 合成样本:SMOTE;

6. ### 选择其他模型:决策树等; 

7. ### 加权少类别人样本错分代价;

8. ### 创新:

   a)  将大类分解成多个小类;

   b)  将小类视为异常点，并用异常检测建模。

## 6.降维

1. ### PCA/ LDA/ ICA；

2. ### 特征选择也是一种降维。

## 代码实战

删除异常值

```python
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
```

```python
train=outliers_proc(train_data,'power',scale=3)
```

```
Delete number is: 963
Now column number is: 149037
Description of data less than the lower bound is:
count    0.0
mean     NaN
std      NaN
min      NaN
25%      NaN
50%      NaN
75%      NaN
max      NaN
Name: power, dtype: float64
Description of data larger than the upper bound is:
count      963.000000
mean       846.836968
std       1929.418081
min        376.000000
25%        400.000000
50%        436.000000
75%        514.000000
max      19312.000000
Name: power, dtype: float64
```

![outliners_pro](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/outliners_pro.png)

特征构造

```python
train['train'] = 1
test_data['train'] = 0
data = pd.concat([train, test_data], ignore_index=True, sort=False)
```

```python
  # 计算特征缺失比例
(data.isnull().sum() / len(data)).sort_values()

```

```
SaleID               0.000000
v_14                 0.000000
v_13                 0.000000
v_12                 0.000000
v_11                 0.000000
v_10                 0.000000
v_9                  0.000000
v_8                  0.000000
v_7                  0.000000
v_6                  0.000000
v_5                  0.000000
v_4                  0.000000
v_3                  0.000000
v_2                  0.000000
v_1                  0.000000
train                0.000000
v_0                  0.000000
creatDate            0.000000
offerType            0.000000
seller               0.000000
regionCode           0.000000
notRepairedDamage    0.000000
kilometer            0.000000
power                0.000000
brand                0.000000
regDate              0.000000
name                 0.000000
model                0.000005
bodyType             0.029678
gearbox              0.039510
fuelType             0.057904
used_time            0.075725
price                0.251210
dtype: float64
```

看一下空数据，有 15k 个样本的时间是有问题的，我们可以选择删除，也可以选择放着但是这里不建议删除，因为删除缺失数据占总样本量过大，7.5%
我们可以先放着，因为如果我们 XGBoost 之类的决策树，其本身就能处理缺失值，所以可以不用管；

```python
# 从邮编中提取城市信息，因为是德国的数据，所以参考德国的邮编，相当于加入了先验知识
data['city'] = data['regionCode'].apply(lambda x: str(x)[:-3])

```

```python
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
    info['brand_price_average'] = round(kind_data['price'].average(), 2)
    all_info[kind] = info

brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"indesx": "brand"})
data = data.merge(brand_fe, how='left', on='brand')
```

**数据分箱**

以 power 为例
这时候我们的缺失值也进桶了，
为什么要做数据分桶呢，原因有很多，=

1. 离散后稀疏向量内积乘法运算速度更快，计算结果也方便存储，容易扩展；
2. 离散后的特征对异常值更具鲁棒性，如 age>30 为 1 否则为 0，对于年龄为 200 的也不会对模型造成很大的干扰；
3. LR 属于广义线性模型，表达能力有限，经过离散化后，每个变量有单独的权重，这相当于引入了非线性，能够提升模型的表达能力，加大拟合；
4. 离散后特征可以进行特征交叉，提升表达能力，由 M+N 个变量编程 M*N 个变量，进一步引入非线形，提升了表达能力；
5. 特征离散后模型更稳定，如用户年龄区间，不会因为用户年龄长了一岁就变化

当然还有很多原因，LightGBM 在改进 XGBoost 时就增加了数据分箱，增强了模型的泛化性

```python
bin = [i * 10 for i in range(31)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data[['power_bin', 'power']].head()
```

|      | power_bin | **power** |
| :--: | :-------: | :-------: |
|  0   |    5.0    |    60     |
|  1   |    NaN    |     0     |
|  2   |   16.0    |    163    |
|  3   |   19.0    |    193    |
|  4   |    6.0    |    68     |

```python
data = data.drop(['creatDate', 'regDate', 'regionCode'], axis=1)
print(data.columns)
```

```
Index(['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType',
       'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode',
       'seller', 'offerType', 'creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3',
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
       'v_13', 'v_14', 'train', 'used_time', 'city', 'brand_amount_x',
       'brand_price_average_x', 'brand_price_max_x', 'brand_price_median_x',
       'brand_price_min_x', 'brand_price_std_x', 'brand_price_sum_x',
       'brand_amount_y', 'brand_price_average_y', 'brand_price_max_y',
       'brand_price_median_y', 'brand_price_min_y', 'brand_price_std_y',
       'brand_price_sum_y', 'power_bin'],
      dtype='object')
```

```python
# 目前的数据其实已经可以给树模型使用了，所以我们导出一下
#data.to_csv('data_for_tree.csv', index=0)
# 数据压缩
data.to_csv(data_for_tree.gz', index=0, compression='gzip', index=False)
```

我们可以再构造一份特征给 LR NN 之类的模型用
之所以分开构造是因为，不同模型对数据集的要求不同
我们看下数据分布：

```python
data['power'].plot.hist()
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/data_power.png)

我们刚刚已经对 train 进行异常值处理了，但是现在还有这么奇怪的分布是因为 test 中的 power 异常值，
所以我们其实刚刚 train 中的 power 异常值不删为好，可以用长尾分布截断来代替

```python
train['power'].plot.hist()
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/train_power.png)

我们对其取 log，在做归一化

```python
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data['power'] = np.log(data[
    'power'] + 1)
data['power'] = ((data['power'] - np.min(data['power'])) /
                 (np.max(data['power']) - np.min(data['power'])))
data.power.plot.hist()
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/minmax_power.png)

```python
data['kilometer'].plot.hist() 
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/kilometer.png)

kilometer 的比较正常，可以直接归一化处理

```python
# 所以我们可以直接做归一化
data['kilometer'] = ((data['kilometer'] - np.min(data['kilometer'])) / 
                        (np.max(data['kilometer']) - np.min(data['kilometer'])))
data['kilometer'].plot.hist()
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/kilometer_scaler.png)

除此之外 还有我们刚刚构造的统计量特征：

```
'brand_amount', 'brand_price_average', 'brand_price_max','brand_price_median', 'brand_price_min', 'brand_price_std','brand_price_sum'
```

```python
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
```

对类别特征进行 OneEncoder

```python
# 对类别特征进行 OneEncoder
data=pd.get_dummies(data,columns=['model','brand_price','brand','fuelType','gearbox','notRepairedDamage','power_bin'])
```

```python
# 这份数据可以给 LR 用
data.to_csv('data_for_lr.csv', index=0)
```

### 特征筛选

#### 特征重要性排名-- Yellowbrick特征可视化库

#### 自动特征筛选工具

#### 过滤式

```python
# 相关性分析
print(data['power'].corr(data['price'], method='spearman'))
print(data['kilometer'].corr(data['price'], method='spearman'))
print(data['brand_amount'].corr(data['price'], method='spearman'))
print(data['brand_price_average'].corr(data['price'], method='spearman'))
print(data['brand_price_max'].corr(data['price'], method='spearman'))
print(data['brand_price_median'].corr(data['price'], method='spearman'))
```

```
0.5737373458520139
-0.4093147076627742
0.0579639618400197
0.38587089498185884
0.26142364388130207
0.3891431767902722
```

相关性热力图

```python
data_numeric = data[['power', 'kilometer', 'brand_amount', 'brand_price_average', 
                     'brand_price_max', 'brand_price_median']]
correlation = data_numeric.corr()

f , ax = plt.subplots(figsize = (7, 7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/相关性热力图.png)

#### 包裹式

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
           k_features=10,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)
x = data.drop(['price'], axis=1)
x = x.fillna(0)
y = data['price']
sfs.fit(x, y)
sfs.k_feature_names_ 
```

```
STOPPING EARLY DUE TO KEYBOARD INTERRUPT...




('powerPS_ten',
 'city',
 'brand_price_std',
 'vehicleType_andere',
 'model_145',
 'model_601',
 'fuelType_andere',
 'notRepairedDamage_ja')
```

```python
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.grid()
plt.show()
```

![](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/plot_sfs.png)

#### 嵌入式



## 7.经验总结

特征工程的主要目的还是在于将数据转换为能更好地表示潜在问题的特征，从而提高机器学习的性能。比如，异常值处理是为了去除噪声，填补缺失值可以加入先验知识等。

特征构造也属于特征工程的一部分，其目的是为了增强数据的表达。

有些比赛的特征是匿名特征，这导致我们并不清楚特征相互直接的关联性，这时我们就只有单纯基于特征进行处理，比如装箱，groupby，agg 等这样一些操作进行一些特征统计，此外还可以对特征进行进一步的 log，exp 等变换，或者对多个特征进行四则运算（如上面我们算出的使用时长），多项式组合等然后进行筛选。由于特性的匿名性其实限制了很多对于特征的处理，当然有些时候用 NN 去提取一些特征也会达到意想不到的良好效果。

对于知道特征含义（非匿名）的特征工程，特别是在工业类型比赛中，会基于信号处理，频域提取，丰度，偏度等构建更为有实际意义的特征，这就是结合背景的特征构建，在推荐系统中也是这样的，各种类型点击率统计，各时段统计，加用户属性的统计等等，这样一种特征构建往往要深入分析背后的业务逻辑或者说物理原理，从而才能更好的找到 magic。

当然特征工程其实是和模型结合在一起的，这就是为什么要为 LR NN 做分桶和特征归一化的原因，而对于特征的处理效果和特征重要性等往往要通过模型来验证。

总的来说，特征工程是一个入门简单，但想精通非常难的一件事。