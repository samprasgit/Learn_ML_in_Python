#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   300 最长递增子序列.py
@Time    :   2021/04/03 20:09:24
@Author  :   Samprasgit 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
class Solution:
    def lengthOfLIST(self,nums):
        """
        动态规划
        时间复杂度：
        空间复杂度：
        """
        if not nums:
            return 0 
        dp=[1]*len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] <nums[i]:
                    dp[i]=max(dp[i],dp[j]+1)

        return max(dp)


    def lengthOfLIST2(self,nums):
        """
        """
        

        
