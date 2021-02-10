#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/9 3:52 下午
# Author : samprasgit
# desc :

from collections import Counter


class Solution:
    def minWindow(self, s, t):
        """
        滑动窗口
        """
        counter = Counter(t)
        left, right = 0, 0
        length = float("inf")
        sums = len(t)
        res = ""
        while right < len(s):
            if counter[s[right]] > 0:
                sums -= 1
            counter[s[right]] -= 1
            right += 1
            while sums == 0:
                if length > right - left:
                    length = right - left
                    res = s[left:right]
                if counter[s[left]] == 0:
                    sums += 1
                counter[s[left]] += 1
                left += 1

        return res
