#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/13 11:43 上午
# Author : samprasgit
# desc :

class Solution:
    def findDisappearedNumbers(self, nums):
        """
        快速遍历
        时间复杂度：O(N)
        空间复杂度：O(N)
        """
        counter = set(nums)
        n = len(nums)
        res = []
        for i in range(1, n + 1):
            if i not in counter:
                res.append(i)
        return res

    def findDisappearedNumbers1(self, nums):
        """
        原地修改数组
        时间复杂度：O(N)
        空间复杂度：O(N)
        """
        for i, num in enumerate(nums):
            if nums[abs(num) - 1] > 0:
                nums[abs(num) - 1] *= -1
        res = []
        for i in range(len(nums)):
            if nums[i] > 0:
                res.append(i + 1)
        return res


if __name__ == "__main__":
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    print(Solution().findDisappearedNumbers1(nums))
