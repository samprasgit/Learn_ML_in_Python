

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

