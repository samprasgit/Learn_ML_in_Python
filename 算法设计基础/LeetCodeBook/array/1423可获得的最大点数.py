#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# 滑动窗口
class Solution1:
    def maxScore(self, cardPoints):
        n = len(cardPoints)
        window_size = n - k
        sums = 0
        res = float("inf")
        for i in range(n):
            sums += cardPoints[i]
            if i >= window_size:
                sums -= cardPoints[i - window_size]
            if i >= window_size - 1:
                res = min(res, sums)

        return sum(cardPoints) - res


# 前缀和
class Solution2:
    def masxScore(self, cardPoints):
        n = len(cardPoints)
        preSum = [0] * (n + 1)
        for i in range(n):
            preSum[i + 1] = preSum[i] + cardPoints[i]

        res = float("inf")
        window_size = n - k
        for i in range(k + 1):
            res = min(res, preSum[i + window_size] - preSum[i])
        return preSum[n] - res
