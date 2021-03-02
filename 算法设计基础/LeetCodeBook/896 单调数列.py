#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/28 1:29 下午

class Solution:
    def isMonotonic1(self, A):
        """
        遍历两次  分别判断
        时间复杂度：O(N)
        空间复杂度：O(1)
        """
        return self.isIncreasing(A) or self.isDecreasing(A)

    def isIncreasing(self, A):
        n = len(A)
        for i in range(n - 1):
            if A[i + 1] - A[i] < 0:
                return False
        return True

    def isDecreasing(self, A):
        n = len(A)
        for i in range(n - 1):
            if A[i + 1] - A[i] > 0:
                return False

        return False

    def isMonotonic2(self, A):
        """
        一次遍历
        空间复杂度：O(N)
        时间复杂度：O(1)
        """
        n = len(A)
        inc, dec = True, True
        for i in range(1, n):
            if A[i] < A[i - 1]:
                inc = False
            if A[i] > A[i - 1]:
                dec = False
            if not inc and not dec:
                return False
        return True
