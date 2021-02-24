#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/22 1:32 下午
class Solution:
    def isToeplitzMatrix(self, matrix):
        """
        每个元素与其右下角元素相等
        """
        for i in range(len(matrix) - 1):
            if matrix[i][:-1] != matrix[i + 1][1:]:
                return False

        return True

    def isToeplitzMatrix1(self, matrix):
        """
        直接遍历
        """
        m = len(matrix)
        n = len(matrix[0])
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] != matrix[i - 1][j - 1]:
                    return False
        return True
