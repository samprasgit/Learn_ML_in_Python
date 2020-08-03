# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-08-02 11:24:22
# 描述: ；连续子数组的最大和


class Solution:

    def maxSubArray(nums):
        for i in range(1, len(nums)):
            nums[i] += max(nums[i - 1], 0)
        return max(nums)
