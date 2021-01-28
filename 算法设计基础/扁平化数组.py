#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/28 10:32 上午
# Author : samprasgit
# desc : 多层嵌套列表扁平化


## 两层
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(data)
print([num for row in data for num in row])

# https://www.sohu.com/a/329517998_797291
