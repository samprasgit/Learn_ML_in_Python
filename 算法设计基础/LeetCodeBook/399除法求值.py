#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/6 4:31 下午
# Author : samprasgit
# desc : 除法求值

class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}

    def find(self, x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] != None:
            root = self.father[root]
        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father

        return root

    def merge(self, x, y, val):
        """
        合并两个节点
        """
        root_x, root_y = self.find(x), self.find(y)

        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connectd(self, x, y):
        """
        判断两个节点是否相连
        """
        return self.father[x] == self.father[y]

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
