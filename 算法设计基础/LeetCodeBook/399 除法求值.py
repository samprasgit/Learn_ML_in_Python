#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/1/6 6:22 下午
# Author : samprasgit
# desc :

#  带权重并查集
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        记录每个节点到根节点的权重
        """
        self.father = {}
        self.value = {}

    def find(self, x):
        """
        查找根节点
        路径压缩
        更新权重
        """
        root = x
        # 节点更新权重时要放大的倍数
        base = 1

        while self.father[root] != None:
            root = self.father[root]
            base *= self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            #  节点离根节点越远，放大的倍数越高
            self.value[x] *= base
            base /= self.father[original_father]

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
        # 更新根节点的权重
        self.value[root_x] = self.value[y] * val / self.value[x]

    def is_connectd(self, x, y):
        """
        判断两个节点是否相连
        """
        return x in self.value and y in self.value and self.father[x] == self.father[y]

    def add(self, x):
        """
        添加新节点
        初始化权重为1.0
        """
        if x not in self.father:
            self.father[x] = None
            self.value[x] = 1.0


class Solution1:
    def calcEquation(self, equations, values, queries):
        uf = UnionFind()
        for (a, b), val in zip(equations, values):
            uf.add(a)
            uf.add(b)
            uf.merge(a, b, val)

        res = [-1.0] * len(queries)
        for i, (a, b) in enumerate(queries):
            if uf.is_connectd(a, b):
                res[i] = uf.value[a] / uf.value[b]

        return res


if __name__ == "__main__":
    equations = [["a", "b"], ["b", "c"]]
    values = [2.0, 3.0]
    queries = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]
    s = Solution1()
    print(s.calcEquation(equations, values, queries))
