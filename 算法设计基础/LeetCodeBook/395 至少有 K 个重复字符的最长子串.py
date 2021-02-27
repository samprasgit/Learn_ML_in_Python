#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/27 3:59 下午

class Solution:
    def longestSubstring(self, s, k):
        """
        递归
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if len(s) < k:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))

        return len(s)
