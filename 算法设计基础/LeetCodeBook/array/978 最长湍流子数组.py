#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time : 2021/2/8 10:24 下午
# Author : samprasgit
# desc :


class Solution:
    def maxTurbulenceSize(self, arr):
        """
        动态规划
        定义两个状态数组  -- 降低空间复杂度
        """
        n = len(arr)
        up = [1] * n
        down = [1] * n
        #  最小为1
        res = 1
        for i in range(1, n):
            if arr[i - 1] < arr[i]:
                up[i] = down[i - 1] + 1
            elif arr[i - 1] > arr[i]:
                down[i] = up[i - 1] + 1

            res = max(res, max(up[i], down[i]))

        return res

    def maxTurbulenceSize2(self, arr):
        """
        滑动窗口
        """
        n = len(arr)
        left = right = 0
        res = 0
        while right < n - 1:
            if left == right:
                if arr[left] == arr[left + 1]:
                    left += 1
                right += 1
            else:
                if arr[right - 1] < arr[right] and arr[right] > arr[right + 1]:
                    right += 1
                elif arr[right - 1] > arr[right] and arr[right] < arr[right + 1]:
                    right += 1
                else:
                    left = right

            res = max(res, right - left + 1)

        return res

    def maxTurbulenceSize2(self, arr):
        """
        双指针
        """

