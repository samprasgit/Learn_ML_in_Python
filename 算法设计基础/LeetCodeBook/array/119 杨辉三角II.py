#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/13 8:20 上午
# Author : samprasgit
# desc :


class Solution:
    def getRow1(self, rowIndex):
        """
        二维数组
        空间复杂度：O(k*(k+1)/2)
        时间复杂度：O(k^2)
        """
        res = [[1 for j in range(i + 1)] for i in range(rowIndex + 1)]
        for i in range(rowIndex + 1):
            for j in range(1 + i):
                res[i][j] = res[i - 1][j - 1] + res[i - 1][j]

        return res[-1]

    def getRow2(self, rowIndex):
        res = [1] * (rowIndex + 1)
        for i in range(2, rowIndex + 1):
            for j in range(i - 1, 0, -1):
                res[j] += res[j - 1]

        return res
