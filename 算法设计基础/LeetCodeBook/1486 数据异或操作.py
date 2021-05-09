# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   1486 数据异或操作.py
@Time    :   2021/05/07 21:07:17
@Desc    :   连续异或操作
"""


class Solution:
    def xorOperation(self, n, start):
        res = 0
        for i in range(n):
            res ^= (start+2*i)

        return res

    def xorOperation2(self, n, start):
        # 数学方法
