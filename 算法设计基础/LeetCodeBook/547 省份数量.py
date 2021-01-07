#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/7 10:32 上午
# Author : samprasgit
# desc : 省份数量

# 基于并查集
class UnionFind:
    def __init__(self):
        self.father = {}
        # 额外记录集合的数量
        self.num_of_sets = 0

    def find(self, x):
        root = x
        while self.father[root] != None:
            root = self.father[root]

        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father

        return root

    def merge(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y
            # 集合数量-1
            self.num_of_sets -= 1

    def add(self, x):
        if x not in self.father:
            self.father[x] = None
            # 集合数量 +1
            self.num_of_sets += 1


class Solution:
    def findCircleNum(self, isConnected):
        """
        并查集
        """
        uf = UnionFind()
        for i in range(len(isConnected)):
            uf.add(i)
            for j in range(i):
                if isConnected[i][j]:
                    uf.add(j)
                    uf.merge(i, j)

        return uf.num_of_sets
