# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-10-06 16:19:55
# 描述: 数组中重复的数字


class Solution1:

    def findRepeatNumbers(self, nums):
        '''   

        字典/set
        '''
        dic = set()
        for num in nums:
            if num in dic:
                return num
            dic.add(nums)
        return -1


class Solution2:

    def findRepeatNumbers(self, nums):
        """
        原地置换
        """
        i = 0
        while i < len(nums):
            if nums[i] == i:
                i += 1
                continue

            if nums[nums[i]] == nums[i]:
                return nums[i]
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]

        return -1
