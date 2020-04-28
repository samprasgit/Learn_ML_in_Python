# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-27 09:51:51
# 描述: 给定我们一个数字，让我们求子数组之和大于等于给定值的最小长度


class Solution:

    def minSubArray(self, s, nums):
        total, left = 0, 0
        result = len(nums) + 1
        for right, n in enumerate(nums):
            total += n
            while total >= s:
                result = min(result, right - left + 1)
                total -= nums[left]
                left += 1

        return result if result <= len(nums) else 0
