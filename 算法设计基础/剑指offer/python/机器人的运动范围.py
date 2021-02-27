#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/25 12:59 下午
# 搜索 回溯问题

class Solution:
    def movingCount(self, m, n, k):
        """
        DFS
        """

        def dfs(i, j, si, sj):
            if i >= m or j >= n or k < si + sj or (i, j) in visited: return 0
            visited.add((i, j))
            return 1 + dfs(i + 1, j, si + 1 if (i + 1) % 10 else si - 8, sj) + dfs(i, j + 1, si, sj + 1
            if (j + 1) % 10 else sj - 8)

        visited = set()
        return dfs(0, 0, 0, 0)

    def movingCount2(self, m, n, k):
        """
        BFS
        """
        queue, visited = [0, 0, 0, 0], set()
        while queue:
            i, j, si, sj = queue.pop()
            if i >= m or j >= n or k < si + sj or (i, j) in visited: continue
            visited.add((i, j))
            queue.appned((i + 1, j, si + 1 if (i + 1) % 10 else si - 8, sj))
            queue.append((i, j + 1, si, sj + 1 if (j + 1) % 10 else sj - 8))

        return len(visited)
