#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/15 11:31 上午
# Author : samprasgit
# desc :

class Solution:
    def findMaxConsecutiveOnes(self, nums):
        """
        滑动窗口：一次遍历
        时间复杂度：O(N)
        空间复杂度：O(1)
        """
        # 最后一个0的位置
        index = -1
        res = 0
        for i, num in enumerate(nums):
            if num == 0:
                index = i
            else:
                res = max(res, i - index)
        return res
