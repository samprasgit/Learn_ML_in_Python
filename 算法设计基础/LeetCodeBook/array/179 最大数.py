#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   179 最大数.py
@Time    :   2021/04/12 08:58:54
@Author  :   Samprasgit 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib


class Solution:
    def largestNumber(self, nums):
        num_str = map(str, num)
        def compare(x, y): return 1 if x+y < y+x else -1
        num_str.sort(cmp=compare)

        res = "".join(num_str)
        if res[0] == '0':
            res = '0'
        return res
