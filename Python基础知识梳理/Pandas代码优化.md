<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Python里+ 和join区别](#python%E9%87%8C-%E5%92%8Cjoin%E5%8C%BA%E5%88%AB)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



**参考**

[Pandas常见的性能优化方法](https://zhuanlan.zhihu.com/p/81554435?utm_source=wechat_session&utm_medium=social&utm_oi=36728643518464)

#### Python里+ 和join区别

- 在用"+"连接字符串时，结果会生成新的对象
- 用join时结果只是将原列表中的元素拼接起来，所以join效率比较高

**节省磁盘空间**

> Pandas在保存数据集时，可以对其进行压缩，其后以压缩格式进行读取

 

```python
train.to_csv('random_data.gz', compression='gzip', index=False)
```

gzip压缩文件可以直接读取：

```python
df = pd.read_csv('random_data.gz')
```

