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

