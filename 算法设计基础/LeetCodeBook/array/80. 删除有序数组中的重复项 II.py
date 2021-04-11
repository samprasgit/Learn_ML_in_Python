#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   80. 删除有序数组中的重复项 II.py
@Time    :   2021/04/06 10:01:34
@Author  :   Samprasgit 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib


class Solution:
    def removeDuplicates(self, nums):
        """
        双指针
        """
        slow = 0
        for fast in range(len(nums)):
            if slow < 2 and nums[fast] != nums[slow-2]:
                nums[slow] = nums[fast]
                slow += 1

        return slow
