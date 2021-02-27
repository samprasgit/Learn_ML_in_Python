#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/24 2:38 下午


class Solution:
    def flipAndInvertImage(self, A):
        """
        两次遍历
        时间复杂度：O(2*N^2)
        空间复杂度：O(1)
        """
        rows = len(A)
        cols = len(A[0])
        for row in range(rows):
            A[row] = A[row[::-1]]
            for col in range(cols):
                A[row][col] ^= 1

        return A

    def flipAndInvertImage1(self, A):
        """
        一次遍历
        空间复杂度:O(N^2)
        时间复杂度：O(1)
        """
        n = len(A)
        for i in range(n):
            # 遍历到中间位置
            for j in range((n + 1) // 2):
                A[i][j], A[i][n - 1 - j] = 1 - A[i][n - 1 - j], 1 - A[i][j]

        return A
