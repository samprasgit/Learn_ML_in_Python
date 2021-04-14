#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   263 丑数.py
@Time    :   2021/04/11 11:15:42
@Author  :   Samprasgit
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
# 丑数 就是只包含质因数 2、3 或 5 的正整数。
# for 和 while 的区别


class Solution:
    def isUgly(self, n):
        """
        时间复杂度 O(lgn)
        """
        # 需要注意零和负数
        if n <= 0:
            return False
        factors = [2, 3, 5]
        for factor in factors:
            while n % factors == 0:
                n //= factor

        return n == 1

    # 求第n个丑数
    def nthUglyNumber(self, n):
        """
        动态规划
        O(N)
        """
        if n <= 0:
            return False
        dp = [1]*n
        #  还没有乘过2 3 5的最小丑数
        index2, index3, index5 = 0, 0, 0
        for i in range(1,n):
            dp[i] = min(2*dp[index2], 3*dp[index3], 5*dp[index5])
            if dp[i] == 2*dp[index2]:
                index2 += 1
            if dp[i] == 3*dp[index3]:
                index3 += 1
            if dp[i] == 5*dp[index5]:
                index5 += 1

        return dp[n-1]
