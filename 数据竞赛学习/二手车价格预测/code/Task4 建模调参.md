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

[toc]



## **1** **学习目标** 

- - [x] 了解常用的机器学习模型，并掌握机器学习模型的建模与调参流程
- -[ ] 完成相应学习打卡任务

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

### - 读取数据

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

五折交叉验证

> 在使用训练集对参数进行训练的时候，经常会发现人们通常会将一整个训练集分为三个部分（比如mnist手写训练集）。一般分为：训练集（train_set），验证集（valid_set），测试集（test_set）这三个部分。这其实是为了保证训练效果而特意设置的。其中测试集很好理解，其实就是完全不参与训练的数据，仅仅用来观测测试效果的数据。而训练集和评估集则牵涉到下面的知识了。

>因为在实际的训练中，训练的结果对于训练集的拟合程度通常还是挺好的（初始条件敏感），但是对于训练集之外的数据的拟合程度通常就不那么令人满意了。因此我们通常并不会把所有的数据集都拿来训练，而是分出一部分来（这一部分不参加训练）对训练集生成的参数进行测试，相对客观的判断这些参数对训练集之外的数据的符合程度。这种思想就称为交叉验证（Cross Validation）









</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cv1</th>
      <th>cv2</th>
      <th>cv3</th>
      <th>cv4</th>
      <th>cv5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>0.191642</td>
      <td>0.194986</td>
      <td>0.192737</td>
      <td>0.195329</td>
      <td>0.19445</td>
    </tr>
  </tbody>
</table>
</div>

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

- 《机器学习》 https://book.douban.com/subject/26708119/ (https://book.douban.com/subject/26708119/) 
- 《统计学习方法》 https://book.douban.com/subject/10590856/ (https://book.douban.com/subject/10590856/)
-  《Python大战机器学习》 https://book.douban.com/subject/26987890/ (https://book.douban.com/subject/26987890/)
- 《面向机器学习的特征工程》 https://book.douban.com/subject/26826639/ (https://book.douban.com/subject/26826639/)
-  《数据科学家访谈录》 https://book.douban.com/subject/30129410/ (https://book.douban.com/subject/30129410/)

