# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-09-15 18:44:09
# 描述: 依次给出n个正整数A1，A2，… ，An，将这n个数分割成m段，
# 每一段内的所有数的和记为这一段的权重， m段权重的最大值记为本次分割的权重。
# 问所有分割方案中分割权重的最小值是多少？


# 动态规划
def dp_array(num, n, m):
    dp = [[float('inf') for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(i):
                dp[i][j] = min(dp[i][j], max(dp[k][j - 1], sum(num[k:i])))
    return dp[n][m]
if __name__ == '__main__':
    n, m = list(map(int, input().split()))
    num = list(map(int, input().split()))
    print(dp_array(num, n, m))
