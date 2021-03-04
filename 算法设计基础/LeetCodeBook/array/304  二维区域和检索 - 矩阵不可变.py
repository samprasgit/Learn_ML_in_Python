#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/2 11:42 下午
# 前缀和

class NumMatrix:
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            m, n = 0, 0
        else:
            m, n = len(matrix), len(matrix[0])

        self.preNum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.preNum[i + 1][j + 1] = self.preNum[i + 1][j] + self.preNum[i][j + 1] - self.preNum[i][j] + \
                                            matrix[i][j]

    def sumRegion(self, row1, col1, row2, col2):
        return self.preNum[row2 + 1][col2 + 1] - self.preNum[row2 + 1][col1] - self.preNum[row1][col2 + 1] + \
               self.preNum[row1][col1]
