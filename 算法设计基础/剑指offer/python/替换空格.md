<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [题目描述：](#%E9%A2%98%E7%9B%AE%E6%8F%8F%E8%BF%B0)
- [思路](#%E6%80%9D%E8%B7%AF)
- [Code](#code)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 题目描述：

请实现一个函数，将一个字符串中的空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

## 思路

```
1.用python字符串的replace方法。
2.对空格split得到list，用‘%20’连接（join）这个list
3.由于替换空格后，字符串长度需要增大。先扫描空格个数，计算字符串应有的长度，从后向前一个个字符复制（需要两个指针）。这样避免了替换空格后，需要移动的操作。
```

## Code

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        return s.replace(' ', '%20')

```

```python
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        num_space = 0
        for i in s:
            if i == ' ':
                num_space += 1

        new_length = len(s) + 2 * num_space
        index_origin = len(s) - 1
        index_new = new_length - 1
        new_string = [None for i in range(new_length)]

        while index_origin >= 0 & (index_new > index_origin):
            if s[index_origin] == ' ':
                new_string[index_new] = '0'
                index_new -= 1
                new_string[index_new] = '2'
                index_new -= 1
                new_string[index_new] = '%'
                index_new -= 1
            else:
                new_string[index_new] = s[index_origin]
                index_new -= 1
            index_origin -= 1
        return ''.join(new_string)
```

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        return '%20'.join(s.split(' '))


```

