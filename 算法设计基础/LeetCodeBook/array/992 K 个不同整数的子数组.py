#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/9 11:30 上午
# Author : samprasgit
# desc :


import collections


class Solution:

    def subarraysWithKDistinct(self, A, K):
        """
        滑动窗口
        """
        return self.atMostK(A, K) - self.atMostK(A, K - 1)

    def atMostK(self, A, K):
        """
        A中由最多k个不同整数组成的子数组个数
        滑动窗口
        """
        n = len(A)
        counters = collections.Counter()
        left, right = 0, 0
        distinct = 0
        res = 0
        while right < n:
            if counters[A[right]] == 0:
                distinct += 1
            counters[A[right]] += 1
            while distinct > K:
                counters[A[left]] -= 1
                if counters[A[left]] == 0:
                    distinct -= 1
                left += 1
            res += right - left + 1
            right += 1

        return res
