#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/25 8:43 上午


class Solution:
    def nthUglyNumber(self, n):
        """
        动态规划
        """
        dp, a, b, c = [1] * n, 0, 0, 0
        for i in range(1, n):
            n2, n3, n5 = dp[a] * 2, dp[b] * 3, dp[c] * 5

            dp[i] = min(n2, n3, n5)
            if dp[i] == n2: a += 1
            if dp[i] == n3: b += 1
            if dp[i] == n5: c += 1

        return dp[-1]
