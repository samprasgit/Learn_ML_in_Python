# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   403 青蛙过河.py
@Time    :   2021/04/29 10:45:29
@Desc    :   None
"""
# 记忆化搜索+二分查找
# 动态规划


class Solution:
    def canCross(self, stones):
        from functools import lru_cache

        @lru_cache(None)
        def dfs(pos, step):
            if pos == stones[-1]:
                return True
            for d in [-1, 0, 1]:
                if step+d > 0 and pos+step+d in set(stones):
                    if dfs(pos+step+d, step+d):
                        return True
            return False
        pos, step = 0, 0
        return dfs(pos, step)

    def canCross2(self, stones):
        from collections import defaultdict
        set_stones = set(stones)
        dp = defaultdict(set)
        dp[0] = {0}
        for s in stones:
            for step in dp[s]:
                for d in [-1, 0, 1]:
                    if step+d > 0 and s+step+d in set_stones:
                        dp[s+step+d].add(step+d)

        return len(dp[stones[-1]]) > 0
