#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/3/1 12:39 下午
# Author : samprasgit
# desc :  前缀和
class NumArray:
    def __init__(self, nums):
        n = len(nums)
        self.preSum = [0] * (n + 1)
        for i in range(n):
            self.preSum[i + 1] = self.preSum[i] + nums[i]

    def sumRange(self, i, j):
        return self.preSum[j + 1] - self.preSum[i]
