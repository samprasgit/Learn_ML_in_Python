#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/30 3:26 下午
# Author : samprasgit
# desc : 二维数组的查找

class Solution:
    def findNumber2DArray(self, matrix, target):
        # 空值情况  []  [[]]
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        n = len(matrix)
        m = len(matrix[0])
        # 右上角开始遍历
        row = 0
        col = m - 1
        while row < n and col >= 0:
            if matrix[row][col] > target:
                col -= 1
            elif matrix[row][col] < target:
                row += 1
            elif matrix[row][col] == target:
                return True
        return False
