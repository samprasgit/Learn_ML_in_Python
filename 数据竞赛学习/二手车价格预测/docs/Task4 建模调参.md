<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [**1** **学习目标**](#1-%E5%AD%A6%E4%B9%A0%E7%9B%AE%E6%A0%87)
- [**2** **内容介绍**](#2-%E5%86%85%E5%AE%B9%E4%BB%8B%E7%BB%8D)
  - [1.线性回归模型: 线性回归对于特征的要求;](#1%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%AF%B9%E4%BA%8E%E7%89%B9%E5%BE%81%E7%9A%84%E8%A6%81%E6%B1%82)
  - [2.模型性能验证:](#2%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E9%AA%8C%E8%AF%81)
  - [3.嵌入式特征选择:](#3%E5%B5%8C%E5%85%A5%E5%BC%8F%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9)
  - [4.模型对比:](#4%E6%A8%A1%E5%9E%8B%E5%AF%B9%E6%AF%94)
  - [5.模型调参:](#5%E6%A8%A1%E5%9E%8B%E8%B0%83%E5%8F%82)
- [**3** **相关原理介绍与推荐**](#3-%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86%E4%BB%8B%E7%BB%8D%E4%B8%8E%E6%8E%A8%E8%8D%90)
- [**4** **代码示例**](#4-%E4%BB%A3%E7%A0%81%E7%A4%BA%E4%BE%8B)
  - [- 读取数据](#--%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE)
  - [非线性模型](#%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B)
  - [建模调参](#%E5%BB%BA%E6%A8%A1%E8%B0%83%E5%8F%82)
- [5.参考](#5%E5%8F%82%E8%80%83)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## **1** **学习目标** 

- - [x] 了解常用的机器学习模型，并掌握机器学习模型的建模与调参流程

## **2** **内容介绍**

### 1.线性回归模型: 线性回归对于特征的要求;

- 处理长尾分布
- 理解线性回归模型

### 2.模型性能验证:

- 评价函数与目标函数
- 交叉验证方法;
- 留一验证方法
-  针对时间序列问题的验证
-  绘制学习率曲线
-  绘制验证曲线

### 3.嵌入式特征选择: 

- Lasso回归
- Ridge回归
- 决策树; 

### 4.模型对比:

- 常用线性模型
- 常用非线性模型

### 5.模型调参:

-  贪心调参方法
-  网格调参方法
-  贝叶斯调参方法

## **3** **相关原理介绍与推荐**

- ###  线性回归模型

  https://zhuanlan.zhihu.com/p/49480391

- ### 决策树模型

  https://zhuanlan.zhihu.com/p/65304798

- ###  GBDT模型

  https://zhuanlan.zhihu.com/p/45145899

- ### XGBoost模型

  https://zhuanlan.zhihu.com/p/86816771

- ###  LightGBM模型

  https://zhuanlan.zhihu.com/p/89360721

## **4** **代码示例**

#### 4.1 读取数据

reduce_mem_usage 函数通过调整数据类型，帮助我们减少数据在内存中占用的空间

```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
```

```python
sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))
```

```
Memory usage of dataframe is 60507328.00 MB
Memory usage after optimization is: 15724107.00 MB
Decreased by 74.0%
```

#### 4.2 五折交叉验证的线性回归  Cross Validation

```python
# 缺失值处理
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
# 数据类型转换
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
# 训练集数据划分
train = sample_feature[continuous_feature_names + ['price']]
train_X = train[continuous_feature_names]
train_y = train['price']
```

```python
# 简单建模
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
```

查看训练的线性回归模型的截距（intercept）与权重(coef)

```python
'intercept:'+ str(model.intercept_)

sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```

```
[('v_6', 3364722.3151564174),
 ('v_8', 699921.8482887275),
 ('v_9', 170458.21023140344),
 ('v_7', 32139.09132280064),
 ('v_12', 20714.91355866134),
 ('v_3', 18030.055035382895),
 ('v_11', 11595.195783621277),
 ('v_13', 11354.642135617458),
 ('v_10', 2657.748631767479),
 ('gearbox', 882.1038583282956),
 ('fuelType', 364.5264412196498),
 ('bodyType', 189.54121712220703),
 ('power', 28.561921245758583),
 ('brand_price_median_x', 0.255292407469609),
 ('brand_price_median_y', 0.2552924074667841),
 ('brand_price_std_x', 0.22524975066406236),
 ('brand_price_std_y', 0.22524975064836486),
 ('used_time', 0.14124619493944898),
 ('creatDate', 0.07945640772099297),
 ('brand_amount_y', 0.07442242106242544),
 ('brand_amount_x', 0.07442242106229742),
 ('regionCode', 0.05233990206739235),
 ('regDate', 0.005310934723489556),
 ('brand_price_max_y', 0.0015900291401336854),
 ('brand_price_max_x', 0.0015900291396645744),
 ('SaleID', 5.4061272118415274e-05),
 ('seller', 2.2907042875885963e-06),
 ('offerType', 2.3044412955641747e-07),
 ('train', -8.009374141693115e-08),
 ('brand_price_sum_y', -1.0875121834094359e-05),
 ('brand_price_sum_x', -1.0875121834223323e-05),
 ('name', -0.0002993688063223507),
 ('brand_price_average_x', -0.20271245299857915),
 ('brand_price_average_y', -0.20271245299859728),
 ('brand_price_min_y', -1.122244459847303),
 ('brand_price_min_x', -1.1222444598473038),
 ('city', -6.976630960502906),
 ('power_bin', -34.40153519461272),
 ('v_14', -282.53973241865395),
 ('kilometer', -372.91194115761215),
 ('notRepairedDamage', -495.8428162130516),
 ('v_0', -2059.122111733545),
 ('v_5', -12305.51909556401),
 ('v_4', -15197.378550338892),
 ('v_2', -26315.17228158719),
 ('v_1', -45573.21795387407)]
```

```python
from matplotlib import pyplot as plt'
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()
```

![v9-plt](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/v9-plt.png)

绘制特征v_9的值与标签的散点图，图片发现模型的预测结果（蓝色点）与真实标签（黑色点）的分布差异较大，且部分预测值出现了小于0的情况，说明我们的模型存在一些问题

通过作图我们发现数据的标签（price）呈现长尾分布，不利于我们的建模预测

![train_y_plt](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/train_y_plt.png)

所以对标签进行了 $log(x+1)$ 变换，使标签贴近于正态分布

```python
train_y_ln = np.log(train_y + 1)
```

```python
print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.subplot(1,2,2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
```

![train_y_ln](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/train_y_ln.png)

重新建模，查看训练的线性回归模型的截距（intercept）与权重(coef)

```python
model = model.fit(train_X, train_y_ln)

print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```



```
intercept:-271.37735295214725
[('v_9', 8.021794015956836),
 ('v_5', 5.787436813423152),
 ('v_12', 1.6346495228937958),
 ('v_1', 1.49326858046208),
 ('v_11', 1.1683103471689507),
 ('v_13', 0.9389116326754223),
 ('v_7', 0.7095062057668579),
 ('v_3', 0.6915362107822818),
 ('v_0', 0.009666562073884116),
 ('power_bin', 0.008523141634245777),
 ('gearbox', 0.007867922542659138),
 ('fuelType', 0.006516574815676286),
 ('bodyType', 0.00454111461074244),
 ('city', 0.0031322040273791658),
 ('power', 0.0007110695883160756),
 ('brand_price_min_x', 1.6252132274579378e-05),
 ('brand_price_min_y', 1.625213227457927e-05),
 ('creatDate', 1.5846788789649107e-05),
 ('brand_amount_y', 1.4474062921760455e-06),
 ('brand_amount_x', 1.447406292173844e-06),
 ('brand_price_median_x', 6.776900622986613e-07),
 ('brand_price_median_y', 6.776900622784534e-07),
 ('brand_price_std_y', 3.838004935782697e-07),
 ('brand_price_std_x', 3.838004935630302e-07),
 ('brand_price_max_y', 3.099649306463753e-07),
 ('brand_price_max_x', 3.0996493061511797e-07),
 ('brand_price_average_x', 2.5294747354086086e-07),
 ('brand_price_average_y', 2.5294747353975405e-07),
 ('SaleID', 2.118442554323473e-08),
 ('seller', 6.988187806200585e-11),
 ('train', -4.320099833421409e-12),
 ('brand_price_sum_y', -7.611244978221208e-11),
 ('brand_price_sum_x', -7.611244978388074e-11),
 ('offerType', -1.738200694489933e-10),
 ('name', -7.027793641666465e-08),
 ('regDate', -1.4601440543007201e-06),
 ('regionCode', -5.378062375072283e-06),
 ('used_time', -4.363732025020281e-05),
 ('v_14', -0.003633396693503721),
 ('kilometer', -0.013832372455444119),
 ('notRepairedDamage', -0.2700348751516212),
 ('v_4', -0.8284609990349868),
 ('v_2', -0.9632797963981975),
 ('v_10', -1.604298529145125),
 ('v_8', -40.22978897413428),
 ('v_6', -238.22531407704605)]
```

再次进行可视化，发现预测结果与真实值较

![img](data:image/png;base64,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2mHz1kRFvb9EJFLNzbBhQ/Djn0vvr9ZWGNncsWtXUJ4aZN/QMPGeXi+//PLQxFjveMc7OOecc3j66adJJBIsW7YMgHvvvZdHH32Uo48+GggSOR511FH84Q9/YN68eRx00EEAfOxjH8s4t8mvfvUrbrzxRiBow9lvv/144YUXhq1z++23c/vtt3PYYcFEuDt37mTz5s28+OKLfPCDHxzKe/b+978/63v59re/TUdHB/vuuy9r1qzBwqh92mmnDaXQH69eN910Exs3buTwww8f+vu87nWvy+VPOSEKKiISuenTIdfUX9naYCab+T7VpjJSekp6d+eEE07ghz/84bB1ospInHqNL3/5y5x33nnDyv/t3/4t531cdNFFo6ZDhomn11+5ciVf//rXc94mH7r8JSJFla0NpgCZ71m2bBn33HPP0MyML730En/84x9529veRnd3N0888QTAqKCTctxxx3H11VcDwcRaf/nLX9h333158cUXh9ZZsWIF11133VBbzbZt23juued45zvfyU9/+lNefvllXnzxRX7+859H9r4y1eu4447jlltu4bnnngOClPs9qQE/EVJQEZGiamuDEZnvqa8PyuM2a9Ysrr/+ek4//XQWLlw4dOlr2rRptLe38773vY8lS5ZkvUx0xRVXcNddd3HooYeydOlSHn30URoaGjj66KNZsGABX/jCF1i+fDlnnHEGRx11FIceeiinnnoqL774IkuWLOEjH/kIixYt4qSTThq6LBWFTPWaP38+X/va11i+fDkLFy7khBNO4JlnnonsNVOU+l5EIjfx1PdBG8qWLcEZSlvbnvYUKTylvheRstbcrCBSKXT5S0REIqMzFRGJhbsPdX0FZRouF5NtEontTMXMDjSzu8zsUTPbZGYXhuWXmtk2M+sMl/embfNlM+sys8fNbEVa+YlhWZeZXZxWPs/M7gvL15jZOIkbRKQQpk2bRl9f39APVCrTcFRJJiUe7k5fXx/Tpk3Lex9xnqn0A3/v7g+Y2b7ARjO7I3zu2+4+LHOamc0HPgocArwJWGdmbwmfvhI4AegF7jezte7+KHBZuK/VZnYNcA5wdYzvSURyMGfOHHp7e9m+fTsAvb2QKSNIXx/MmVPgysmYpk2bxpxJ/FNiCyru/gzwTHj/RTN7DBgrW88pwGp3fxV4ysy6gCPC57rc/UkAM1sNnBLu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)

进行五折交叉验证训练

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
```

```python
# 函数装饰器
def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper
```

使用线性回归模型，对未处理标签的特征数据进行五折交叉验证

```python
scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))
```

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    1.4s finished
```

```python
print('AVG:', np.mean(scores))
# AVG: 1.3651560942289953
```

使用线性回归模型，对处理过标签的特征数据进行五折交叉验证

```python
scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
```

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    1.4s finished
```

```python
print('AVG:', np.mean(scores))
# AVG: 0.1931621754659293 
```

```python
scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
scores
```

</style>

|      |   cv1    |   cv2    |   cv3    |   cv4    |   cv5    |
| :--: | :------: | :------: | :------: | :------: | :------: |
| MAE  | 0.190711 | 0.193717 | 0.194039 | 0.191691 | 0.195653 |

</div>

采用时间顺序对数据集进行分隔--选用靠前时间的4/5样本当作训练集，靠后时间的1/5当作验证集，最终结果与五折交叉验证差距不大

```python
# 数据划分
import datetime

sample_feature = sample_feature.reset_index(drop=True)

split_point = len(sample_feature) // 5 * 4

train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)

model = model.fit(train_X, train_y_ln)

mean_absolute_error(val_y_ln, model.predict(val_X))
# 0.19567171875313785
```

绘制学习率曲线

```python
from sklearn.model_selection import learning_curve, validation_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                     train_scores_mean + train_scores_std, alpha=0.1,  
                     color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                     test_scores_mean + test_scores_std, alpha=0.1,  
                     color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
             label="Training score")  
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
             label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt  
```

```python
plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)  
```

![img](data:image/png;base64,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qk6bNk3973//q+7fv19NSEhQ3W63mp6erg4cOLBaY/zkk0/U2bNnq6qqqhkZGepf/vKXGhnnunXr1MTERFVVVXXbtm3qE088USPjVFVVtdvt6lNPPaXefvvt6u+//15j48zPz1cHDBhQaFv//v3VlJQU1e12q48//ri6f/9+9f/+7//UiRMnqqqqqj///LP6xBNPVGuc27ZtU//xj3+oLpdLtVqt6muvvVZjP1NfM2bMUFetWlUjP9N169apY8aMUVVVVbds2aKOHj26Wj7ToGn2iY2NZdGiRd6ff/31V7p37w5A7969+eGHH9i9ezc9e/ZEURSaNm2Ky+UiIyOj2mK88847efbZZ70/6/X6Ghln3759efHFFwE4ceIE9evXr5FxAsyfP5/BgwfTsGFDoGb+vwMcOHCAvLw8HnvsMYYPH87OnTux2+3ExsaiKAo9e/bkxx9/oMjpPgAAB/BJREFULHPqlOqwZcsW2rVrx9NPP80TTzzBLbfcUmM/U49ffvmF33//nb/+9a818jNt2bIlLpcLt9uN1WrFYDBUy2caNMn/jjvu8I4yBlBVFUVRAG2Kiezs7GLTUni2V5ewsDDCw8OxWq2MGTOG5557rkbGCWAwGJg4cSIvvvgid9xxR42Mc+3atdSrV8/7hw018/8dwGw2M2LECN59911mzpzJpEmTsFgsxWIqbeqU6nL+/Hn279/PwoULmTlzJuPGjauxn6nHm2++ydNPP11qTJf7Mw0NDSU9PZ277rqLadOmkZCQUC2faVC1+fvS6QrKvZycHCIjIys19URVO3nyJE8//TRDhgzh3nvv5aWXXqqRcYJWqx43bhwPPfQQNputxsX56aefoigKP/74I8nJyUycOLFQTammxAla7a9FixYoikLLli2JiIjgwoULxWLNz88vdeqU6lC3bl1atWqFyWSiVatWhISE8OeffxaLsyZ8pgBZWVkcPXqUHj16YLVai8VUEz7TZcuW0bNnT8aOHcvJkyf5+9//jsPhKBZnVX+mQVPzL+qaa67xzhm0efNmunXrRnx8PFu2bMHtdnPixAncbjf16tWrtpjOnj3LY489xvjx43nggQdqbJyff/45b775JgAWiwVFUbjuuutqXJwrVqzgww8/ZPny5bRv35758+fTu3fvGhcnwCeffEJSUhIAp06dIi8vj9DQUI4fP46qqmzZssUba2lTp1SHrl278v3336OqqjfOG2+8sUZ+pgA7d+7kpptuArR5xYxGY437TCMjI71JvE6dOjidzmr5uw+qEb5paWm88MILfPzxx/zxxx9MmzYNh8NBq1atmD17Nnq9nkWLFrF582bcbjeTJk2iW7du1Rbf7Nmz+eabb2jVqpV325QpU5g9e3aNijM3N5dJkyZx9uxZnE4nI0eOpHXr1jXu8/SVkJDAjBkz0Ol0NTJOu93OpEmTOHHiBIqiMG7cOHQ6HXPnzsXlctGzZ0+ef/75EqdOad26dbXGumDBArZv346qqjz//PPExMTUyM8U4J133sFgMPDII48AWnKvaZ9pTk4OkydP5syZMzgcDoYPH851110X8M80qJK/EEIITdA2+wghRDCT5C+EEEFIkr8QQgQhSf5CCBGEJPkLIUQQkuQvaoWkpCQSEhK48847ueWWW0hISGDMmDF+nZucnMzrr79e6v7NmzezevXqqgq12qSlpfHQQw9d7jBELSVdPUWtsnbtWo4ePcq4ceMudyiXne+4FSEqKmindxBXhu3bt/Pyyy9jNBp56KGHMJvNrFixwrt/4cKFHD58mFWrVvHKK69w++23Ex8fzx9//EF0dDSLFi3iiy++4OjRowwePJixY8fSuHFjUlNT6dChAzNnziQjI4Nx48Zht9tp2bIl27ZtY926dYXiWL58Of/+979RFIW7776b4cOHM2bMGG6++Wb69+/PkCFDmDNnDgaDgaSkJNxuN1lZWUydOpX4+Hj69etHly5dSElJoUePHmRnZ7Nv3z5atmzJSy+9RGJiIqqqcvLkSXJzc5k/fz4hISHe99+xYwevvPIKer2e5s2bM2vWLIxGY7X9P4jaR5K/qPVsNhtr1qwBYMmSJbz11ltYLBamT5/Oli1baNSokffY1NRU3n//fZo0acLgwYP55ZdfCl3r2LFjvPvuu1gsFvr27cuZM2d4++236dOnD0OHDmXr1q1s3bq10Dm///47X3/9NR999BGKovDII4/Qs2dPZs+ezZAhQ9iyZQuDBg3immuu4euvv2bixIlcddVVfPnll6xdu5b4+HjS09N5//33adCgAd27d2fNmjVMmzaNPn36kJWVBUDz5s2ZP38+3333HS+99BJTp04FtMnqpk2bxkcffUR0dDSvvvoqn332mTQJiTJJ8he1XsuWLb2vo6OjmThxImFhYRw9epTOnTsXOjYqKoomTZoA0KRJk0IT0oE29bdn5sQGDRpgs9k4cuQI9913H0CJw+kPHTrEiRMnvFMIZGZmcvz4cVq1akX//v1ZunQpL7/8MgANGzbkjTfewGw2k5OT432vunXr0rRpU0Cb5bFNmzYAREREeGPs0aMHAF26dGHu3Lne98/IyOD06dM899xzAOTn53PzzTdX6DMUwUeSv6j1PDO0Zmdn89prr7Fp0yYAHn30UYo+0vJMk1uakva3a9eOn3/+mfbt27Nnz55i+1u1akWbNm145513UBSFZcuW0a5dO1JTU/nqq69ISEhg/vz5TJ8+nTlz5vDyyy/TunVrXnvtNdLT0/2KC7S1CLp168ZPP/1E27ZtvdujoqJo3Lgxb7zxBhEREWzYsIHQ0NByryeCmyR/ccUIDw8nPj6e++67j9DQUCIjIzl9+jQxMTGXdN2RI0cyYcIEvvnmGxo2bFhsut+rr76aG2+8kYcffhi73U7Hjh2pX78+CQkJTJ06lW7duvHII4+wfv16+vfvz1NPPUV0dDSNGzfm/PnzfsexefNmNmzYgNvtZt68ed7tOp2OKVOmMGrUKFRVJSwsjAULFlzSv1lc+aS3jxDl+O6774iKiqJjx4788MMPLFmyhA8++KBaY0hMTOTuu++md+/e1fq+/799O7YBGAiBIIhEG18p9I5cwgcOLJmZCohWl8B/Wf5wcc6JqorMjJmJ7v76JHjN8gdYyIcvwELiD7CQ+AMsJP4AC4k/wEIPClVBrNsz/Q8AAAAASUVORK5CYII=)

> yellowbrick

```python
from yellowbrick.model_selection import LearningCurve

viz = LearningCurve(LinearRegression(),
                    param_name="max_depth",
                    param_range=np.arange(1, 11),
                    cv=5,
                    scoring=make_scorer(mean_absolute_error), n_jobs=1)

# Fit and poof the visualizer
viz.fit(train_X[:1000], train_y_ln[:1000])
viz.poof()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAe8AAAFlCAYAAADComBzAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdeXxU1d0/8M+5s2WZhJCQhISdQFhEQFChIkVEbLUgqIgoorYu/WmxoogibmhREa27Yt2tWuRpi0ut2ucBcQERaRABgYSdQEJCyDKZyWz33vP7YzKThGyTZG7Wz7uvvsjce+fcc2ZivvfsQkopQURERB2G0tYZICIioqZh8CYiIupgGLyJiIg6GAZvIiKiDobBm4iIqINh8CYiIupgGLyp3Tp69CjOOOOMNrn3c889h48++ihi6Xm9Xjz77LOYOXMmZsyYgenTp+PVV19FW8zU3LBhAyZPnoxZs2bB4/E0K401a9bg97//fZ3nbrrpJuzbt68lWQzbkCFDMH36dMyYMQMzZ87Er371K1x++eXYsWNHq9y/qdatW4dly5a1dTaoEzC3dQaI2qPbb789YmlJKXHrrbdiwIABWL16NWw2G0pKSvD73/8eFRUVWLBgQcTuFY5///vfuOKKK3Drrbcakv5rr71mSLr1eeedd5CYmBh6/cYbb2DZsmVYvXp1q+YjHFOmTMGUKVPaOhvUCTB4U4fk8/nw1FNPYcuWLdA0DcOHD8f9998Pu92O9evX4y9/+Qt8Ph+Ki4sxc+ZMLFiwAJs3b8ajjz6KmJgYuFwu3H333XjppZfQp08f7N27F6qq4uGHH8bYsWOxePFiDB48GDfccANOP/103Hzzzdi4cSMKCwtx44034uqrr4amaVixYgW+/PJLxMXFYeTIkdi/fz/efffdGnndsmULDhw4gFdffRUmkwkA0L17d6xYsQLHjh0DAMybNw9z587Fr3/961qvR4wYgSlTpmDPnj2YNWsWsrKy8MorrwAA9u/fj+uvvx5fffUVDh06hEcffRSlpaXQNA3z5s3DrFmzauTl9ddfx7p162Cz2VBeXo4777wTy5cvx6ZNm2AymTBy5Ejce++9sNvtOP/88zFy5EhkZ2fjzjvvxNSpU8P6bs4//3w899xzqKiowDPPPFPn5xuJ7++f//xnrXurqor8/Hx069YtdGzlypX43//9X+i6jl69euGhhx5CamoqDh8+jCVLlqCsrAzJycmQUuKSSy7B2Wefjblz5yIjIwPHjh3Du+++i6NHj+Kpp56C2+2GoiiYP38+Jk+ejBMnTuCee+5BSUkJAGDSpElYsGBBvcfXrFmD//znP/jLX/6C48ePY+nSpTh27BiklJg5cyZuvPFGHD16FNdffz0mTZqEn376CQ6HA4sWLQr786cuQhK1U7m5uXL06NF1nnvhhRfk8uXLpa7rUkop//znP8uHHnpI6rour7nmGnnw4EEppZTHjx+Xw4YNkydPnpTff/+9HDp0qDx69KiUUsrvv/9eDhs2TO7atUtKKeUbb7wh586dK6WU8p577pGvv/66lFLKzMxM+e6770oppdyxY4ccMWKE9Hg8ctWqVXLu3LnS4/FIr9crf/e738lrrrmmVl7feOMN+cc//rHBsl5zzTXy888/r/N1Zmam/PDDD6WUUpaXl8szzzxTFhYWSimlXLFihXz66ael3++XF198sdy5c6eUUkqHwyEvuugi+eOPP9a6V/WyPffcc3L+/PnS5/NJTdPk4sWL5QMPPCCllHLy5MnyxRdfrDO///znP+XNN99c57nJkyfL7du3N/j5RuL7C34206ZNk9OmTZMTJkyQ559/vvzTn/4ki4qKpJRSfvjhh3LBggXS7/dLKaX84IMP5I033iillHL27Nny/fffl1JKuW/fPjlq1Cj5z3/+U+bm5srMzEy5ZcsWKaWUpaWl8sILL5S5ubmhPP3yl7+Ux44dky+++GLo83K5XHLBggXS4XDUe7z65zZ37lz55ptvhr6v6dOny08//TR0/y+//FJKKeUXX3whzzvvvDo/a+q6WPOmDumrr75CeXk5vvvuOwCA3+9HUlIShBB45ZVX8NVXX+HTTz/F/v37IaWE2+0GAKSlpaFXr16hdNLT0zFs2DAAwPDhw/Hhhx/Web9gU+dpp50Gn8+HiooKfP3115gxYwZsNhsA4Morr6xV6wYARVFa3Ld95plnAgDsdjumTp2KTz75BNdffz3+9a9/4f3338ehQ4dw5MgRLFmyJPQej8eDXbt2YfTo0fWm+8033+COO+6AxWIBEKjx/+EPf6h13+aq7/ON1PcHVDWb//zzz7j55psxbtw4JCUlAQDWr1+PHTt24PLLLwcA6LoOt9uNsrIybN++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BwA4HI4WfELtV5cdbe72OeHylrV1NoiIQhZPqXtTi3umnGbI/YJbTZaXl+P555/HM888g2XLlsFms0HKmq0AjVV06jqfmZmJH38MNPdv27at1nmfz4cFCxYgPz8fQCDo9+jRA1arFRkZGaHa/ZYtW/Dkk08iIyMD//3vfwEABQUFcDgcSEhIqFGWgQMH4vrrr8e7776LZ599NhTsO5suW/OGBFzeEsTaurH2TUTtQnBQ2hPrfsauglIMT03APVNOa9Fo83DY7XaMGTMGl156KWJiYhAfH4/CwkL07t27RenedNNNuPvuu/H5558jJSWl1pz35ORk3H///Zg/fz7MZjM0TcN5552Hc889F8OHD8eSJUvwySefAAAee+wxxMXFYcmSJfjPf/4Dj8eDRx55pFaa/+///T/cd999+J//+R84nc56R8p3dEKe+njVDgXXeI3k2uYuTxnK3IXoFpOCWFu3Rq/vyOsqh4Pl6/g6exk7YvmCy6KGs+qWy+VCbGys0VlqVV9//TW6d++OkSNH4rvvvsNLL71UY1BcZ9Pc77Cu35PG4l7XrXkDAAScnpKwgjcRETVN7969sWTJktAKYgsXLmzrLHUaXTx4A5ruR4XPgRhrfFtnhYioU8nIyMDq1atDr6vvNEkt02UHrAUJocDpKWnrbBAREYWtywdvAFA1H9w+Z1tng4iIKCwM3gjWvovbOhtERERhYfCu5NM88PjZH0NERO0fg3clRZjgcJ9s62wQURd34MRP+Hjrs3hnwxJ8vPVZHDjxU4vT3Lt3L26++WbMmzcPl19+OZ5//vlai7BE0t13341//OMfNY69/fbbeOmll+p9z4QJEwAAjz76KPLy8mqc279/P+bNm9fgPd977z0AgZ3Oqg+Sa46vv/4a1113HX7729/i2muvDc01b0+6/Gjz6lTNA6+/AjZLTFtnhYi6oAMnfsI32VXrcJdUHA+9Hpg8qllpOhwO3HnnnXjhhRfQv3//0IYeH3zwAa666qqI5PtUs2fPxnPPPRda3hQAPvzwQzz11FONvve+++5r1j1XrlyJa665psvsdMbgXY0QJpR7ihm8icgQWw5+hkNF2+s8J6WE219e57kNOauRdejzOs/17zESZw24uN57rlu3DuPGjUP//v0BACaTCU888QQsFkutncWSk5Px7LPPwmazISEhAY899hhUVcWCBQsgpYTf78fDDz+M/v374/bbb4fT6YTH48GiRYswbty40D3PPPNMFBcX49ixY+jVqxe2b9+OHj16ID09HTk5OVi+fDl0XYfD4cD999+PMWPGhN47b948LF26FHFxcbjrrrsgpURycnLo/BdffFFr97PVq1ejrKwMS5cuxciRI0Nbn7755pv497//DbPZjDPPPBOLFi3CCy+8gKNHj+LkyZPIy8vDvffei4kTJ9b4zJq705nP58Odd97ZKjudMXifwqu64VXdsJmj2zorRNTFSFn37mF6PcfDUVhYiD59+tQ4Vn0VsODOYlJKTJkyBatWrUJqaireeecdrFy5EuPGjUNcXBz+/Oc/Y9++fXA6nThy5AiKiorw9ttv4+TJkzh06FCt+86aNQuffPIJbrnlFqxZswZz5swBAOzbtw/33HMPhgwZgn/9619Ys2ZNjeAd9NZbb2HatGmYPXs2Pvvss9DOYIcOHaq1+9ktt9yC9957D0uXLsWaNWsAANnZ2fj888/xwQcfwGw247bbbsP69esBBFYye/3117Fx40a8+eabtYJ3c3c6O3jwIG644YZW2emMwfsUilDgdBfDFterrbNCRJ3MWQMurreW7HK5sDb7NZRUHK91rntMT8wYs6BZ90xPT8euXbtqHMvNzcXx44H7BHfjKikpgd1uD20DetZZZ+Hpp5/GokWLcOjQIdx6660wm8245ZZbMHjwYMydOxd33nknVFWtsz96xowZuP766/G73/0OP/zwA+6//354vV6kpKTg5ZdfRlRUFFwuV2gnslPt3bsXM2bMAACMGTMmFOwa2/0s6MCBAxg1ahQsFguAQGvA3r17AQDDhg0DAPTs2TO0NGlQS3Y6S0lJabWdzjhgrQ4etQI+1dv4hUREEXR6n8lNOh6OyZMn49tvv8WRI0cAAH6/H8uXL0dOTg6Aqt24unfvDqfTicLCQgDADz/8gP79+2Pz5s1ISUnBm2++iVtuuQVPP/00srOz4XK58Oqrr2L58uX405/+VOu+iYmJyMjIwMsvv4ypU6eGNhB59NFH8cc//hFPPPEEMjMz6x04N3DgwNCOZMHdxRra/ezUdAYOHIjt27dDVVVIKbFly5ZQMG1oM6qW7HRWWFjYajudseZdB0UoKPcUIcnO2jcRtZ7goLQduetR6i5EQnQKTu8zudmD1YDAjmHLly/H/fff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)







</style>

| -    | LinearRegression | DecisionTreeRegressor | RandomForestRegressor | GradientBoostingRegressor | MLPRegressor | XGBRegressor | LGBMRegressor |
| :--- | ---------------- | --------------------: | --------------------: | ------------------------: | -----------: | -----------: | ------------- |
| cv1  | 0.192340         |              0.199679 |              0.142387 |                  0.177219 |  1149.852733 |     0.139966 | 0.146168      |
| cv2  | 0.193093         |              0.186864 |              0.140533 |                  0.177653 |   462.532207 |     0.140220 | 0.146167      |
| cv3  | 0.193228         |              0.186042 |              0.139830 |                  0.177274 |   440.980327 |     0.140638 | 0.145961      |
| cv4  | 0.191755         |              0.185118 |              0.137571 |                  0.176387 |   689.471665 |     0.138488 | 0.143913      |
| cv5  | 0.182370         |              0.189166 |              0.131268 |                  0.165530 |   151.629030 |     0.133750 | 0.135703      |

</div>



### 建模调参

在此我们介绍了三种常用的调参方法如下：

- 手动调参
  - 贪心算法
  - 网格调参
  - 贝叶斯调参 

- 自动调参
  - auto-sklearn
  - Hyperot 
  - 

## 5.参考

- [机器学习](https://book.douban.com/subject/26708119/)
- [统计学习方法-李航](https://book.douban.com/subject/10590856/)
-  [Python大战机器学习]( https://book.douban.com/subject/26987890/)
- [面向机器学习的特征工程]( https://book.douban.com/subject/26826639/)
-  [数据科学家访谈录]( https://book.douban.com/subject/30129410/) 

