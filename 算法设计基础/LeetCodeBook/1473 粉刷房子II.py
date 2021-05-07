# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   1473 粉刷房子II.py
@Time    :   2021/05/04 16:14:23
@Desc    :   粉刷房子  三维动态规划
"""


class Solution:
    def minCost(self, houses, cost, m, n, target):
        houses = [c - 1 for c in houses]

        # dp 所有元素初始化为极大值
        dp = [[[float("inf")] * target for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if houses[i] != -1 and houses[i] != j:
                    continue

                for k in range(target):
                    for j0 in range(n):
                        if j == j0:
                            if i == 0:
                                if k == 0:
                                    dp[i][j][k] = 0
                            else:
                                dp[i][j][k] = min(dp[i][j][k], dp[i - 1][j][k])
                        elif i > 0 and k > 0:
                            dp[i][j][k] = min(
                                dp[i][j][k], dp[i - 1][j0][k - 1])

                    if dp[i][j][k] != float("inf") and houses[i] == -1:
                        dp[i][j][k] += cost[i][j]

        ans = min(dp[m - 1][j][target - 1] for j in range(n))
        return -1 if ans == float("inf") else ans
