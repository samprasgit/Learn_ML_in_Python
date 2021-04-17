#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   213. 打家劫舍 II.py
@Time    :   2021/04/15 11:44:33
@Author  :   Samprasgit 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
# 打家劫舍 I II


class Solution:
    def rob1(self, nums):
        n = len(nums)
        if not nums:
            return 0
        if n == 1:
            return nums[0]
        dp=[0]*n 
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])

        return dp[-1]


    def rob2(self, nums):
        # 首尾相连
        n=len(nums)
        if not nums:
            return 0 
        if n==1: 
            return nums[0]  
        return max(self.rob2(nums[0:n-1]),self.rob2(nums[1:n]))
