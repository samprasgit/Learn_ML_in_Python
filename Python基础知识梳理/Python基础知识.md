[廖雪峰的官方网站](https://www.liaoxuefeng.com/)

#### 使用list和tuple

list是一种有序的集合，可以随时添加和删除其中的元素。

tuple和list非常类似，但是tuple一旦初始化就不能修改

“可变的”tuple：

```python
>>> t = ('a', 'b', ['A', 'B'])
>>> t[2][0] = 'X'
>>> t[2][1] = 'Y'
>>> t
('a', 'b', ['X', 'Y'])
```

tuple所谓的“不变”是说，tuple的每个元素，指向永远不变,但指向的这个list本身是可变的

