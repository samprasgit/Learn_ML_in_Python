# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-23 17:56:34
# 描述: 动态规划例题


class Solution:

    def climbStairs(self, n):
        '''

        70 爬楼梯

        Arguments:
                n {[type]} -- [description]
        '''
        if n < 2:
            return n
        a = 1  # 边界
        b = 2  # 边界
        temp = 0
        for i in range(0, n + 1):
            temp = a + b   # 状态转移
            a = b  # 最优子结构
            b = temp  # 最优子结构

        return temp

    def rob(self, n):
        '''

        198 打家劫舍
        Arguments:
                n {[type]} -- [description]
        '''
