#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/28 10:09 上午
# Author : samprasgit
# desc : 前缀和

class Solution1:
    def pivotIndex(self, nums):
        """
        前缀和
        """
        n = len(nums)
        s = sum(nums)
        t = 0  # 左边求和
        for i in range(n):
            if 2 * t + nums[i] == s:
                return i
            t += nums[i]

        return -1
