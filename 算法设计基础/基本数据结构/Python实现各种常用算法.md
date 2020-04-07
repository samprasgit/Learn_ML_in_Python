<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [基本数据结构与算法](#%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E4%B8%8E%E7%AE%97%E6%B3%95)
- [堆](#%E5%A0%86)
- [栈](#%E6%A0%88)
- [单链表](#%E5%8D%95%E9%93%BE%E8%A1%A8)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 基本数据结构与算法

10个数据结构:数组、链表栈、队列、散列表、二叉树、堆、跳表、图、Trie树;

10个算法:递归、排序分查找、搜索、哈希算法、贪心算法、分治算法、回溯算法、动态规划、字符串匹配算法。

![](/Users/sampras/Desktop/NotebookAlgorithm/算法设计基础/基本数据结构/img/数据结构与算法.jpg)



## 堆

```python
class heap(object):
    def __init__(self):
        #初始化一个空堆，使用数组来在存放堆元素，节省存储
        self.data_list = []
    def get_parent_index(self,index):
        #返回父节点的下标
        if index == 0 or index > len(self.data_list) -1:
            return None
        else:
            return (index -1) >> 1
    def swap(self,index_a,index_b):
        #交换数组中的两个元素
        self.data_list[index_a],self.data_list[index_b] = self.data_list[index_b],self.data_list[index_a]
    def insert(self,data):
        #先把元素放在最后，然后从后往前依次堆化
        #这里以大顶堆为例，如果插入元素比父节点大，则交换，直到最后
        self.data_list.append(data)
        index = len(self.data_list) -1 
        parent = self.get_parent_index(index)
        #循环，直到该元素成为堆顶，或小于父节点（对于大顶堆) 
        while parent is not None and self.data_list[parent] < self.data_list[index]:
            #交换操作
            self.swap(parent,index)
            index = parent
            parent = self.get_parent_index(parent)
    def removeMax(self):
        #删除堆顶元素，然后将最后一个元素放在堆顶，再从上往下依次堆化
        remove_data = self.data_list[0]
        self.data_list[0] = self.data_list[-1]
        del self.data_list[-1]

        #堆化
        self.heapify(0)
        return remove_data
    def heapify(self,index):
        #从上往下堆化，从index 开始堆化操作 (大顶堆)
        total_index = len(self.data_list) -1
        while True:
            maxvalue_index = index
            if 2*index +1 <=  total_index and self.data_list[2*index +1] > self.data_list[maxvalue_index]:
                maxvalue_index = 2*index +1
            if 2*index +2 <=  total_index and self.data_list[2*index +2] > self.data_list[maxvalue_index]:
                maxvalue_index = 2*index +2
            if maxvalue_index == index:
                break
            self.swap(index,maxvalue_index)
            index = maxvalue_index
```

## 栈

class Stack(object):

```python
def __init__(self, limit=10):
    self.stack = []  # 存放元素
    self.limit = limit  # 栈容量极限

def push(self, data):  # 判断栈是否溢出
    if len(self.stack) >= self.limit:
        print('StackOverflowError')
        pass
    self.stack.append(data)

def pop(self):
    if self.stack:
        return self.stack.pop()
    else:
        raise IndexError('pop from an empty stack')  # 空栈不能被弹出

def peek(self):  # 查看堆栈的最上面的元素
    if self.stack:
        return self.stack[-1]

def is_empty(self):  # 判断栈是否为空
    return not bool(self.stack)

def size(self):  # 返回栈的大小
    return len(self.stack)
```
## 单链表