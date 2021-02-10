#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/8 12:38 下午
# Author : samprasgit
# desc : 旋转数组
class Solution:
    def rotate(self, nums, k):
        n = len(nums)
        k %= n
        nums.reverse()
        nums[:k] = list(reversed(nums[:k]))
        nums[k:] = list(reversed(nums[k:]))
        return nums
