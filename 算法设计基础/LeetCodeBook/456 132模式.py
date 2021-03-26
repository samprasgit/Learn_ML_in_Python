#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/24 8:33 上午
class Solution:
    def find132pattern(self, nums):
        """
        枚举
        时间复杂度：O(nlog(n))
        
        """
        from sortedcontainers import SortedList
        n = len(nums)
        if n < 3:
            return False
        # min1  左侧最小值
        min1 = nums[0]
        # 右侧所有元素
        right = SortedList[nums[2:]]

        for j in range(1, n):
            if min1 < nums[j]:
                index = right.bisect_right(min1)
                if index < len(right) and right[index] < nums[j]:
                    return True
            min1 = min(min1, nums[j])
            right.remove(nums[j + 1])

        return False

    def find132pattern2(self, nums):
        """
        单调栈
        时间复杂度：O（N）
        空间复杂度：O（N）
        """

        N = len(nums)
        leftMin = [float("inf")] * N
        for i in range(1, N):
            leftMin[i] = min(leftMin[i - 1], nums[i - 1])

        stack = []
        for j in range(N - 1, -1, -1):
            numsk = float("-inf")
            while stack and stack[-1] < nums[j]:
                numsk = stack.pop()
            if leftMin[j] < numsk:
                return True
            stack.append(nums[j])

        return False
