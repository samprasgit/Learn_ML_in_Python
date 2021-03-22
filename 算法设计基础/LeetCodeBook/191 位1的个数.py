#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/22 8:00 下午
class Solution:
    def hammingWeight(self, n):
        """
        库函数
        """
        return bin(n).count("1")

    def hammingWeight(self, n):
        """
        右移32次
        """
        res = 0
        while n:
            res += n & 1
            n >>= 1

        return res
