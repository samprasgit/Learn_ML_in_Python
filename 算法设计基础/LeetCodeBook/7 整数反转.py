# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   7 整数反转.py
@Time    :   2021/05/03 22:30:52
@Desc    :   整数反转
"""


class Solution:
    def reverse(self, x):
        INT_MIN, INT_MAX = -2**31, 2**31-1
        res = 0
        while x != 0:
            if res < INT_MIN//10+1 or res > INT_MAX//10:
                return 0
            digit = x % 10
            if x < 0 and digit > 0:
                digit -= 10
            x = (x-digit)//10
            res = res*10+digit

        return res
