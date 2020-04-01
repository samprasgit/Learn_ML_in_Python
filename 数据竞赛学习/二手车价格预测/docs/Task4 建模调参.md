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

![v9_ln](/Users/sampras/Desktop/下载的项目/Learn_ML_in_Python/数据竞赛学习/二手车价格预测/img/v9_ln.png)











### 非线性模型





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

