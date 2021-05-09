# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   740 删除获取点数.py
@Time    :   2021/05/07 21:11:39
@Desc    :   None
"""


class Solution:
    def deleteAndEarn(self, nums):
        maxVal = max(nums)
        dp = [0]*(maxVal+1)
        for val in nums:
            dp[val] += val

        def rob(nums):
            size = len(nums)
            first, second = nums[0], max(nums[0], nums[1])
            for i in range(2, size):
                first, second = second, max(first+nums[i], second)

            return second

        return rob(dp)
