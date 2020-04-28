# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-27 09:59:51
# 描述: 给定一个 n × n 的二维矩阵表示一个图像,将图像顺时针旋转 90 度。
# 你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。


class Solution:

    def rotate(self, matrix):
        '''[summary]

        [description]

        Arguments:
                matrix {[type]} -- [description]
        '''
        n = len(matrix[0])
        # 转置
        for i in range(n):
            for j in range(i, n):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]

        for i in range(n):
            matrix[i].reverse()
