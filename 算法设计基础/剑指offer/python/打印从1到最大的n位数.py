#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/2/1 4:32 下午
# Author : samprasgit
# desc : 打印从1到最大的n位数


class Solution:
    def printNumbers(self, n: int) -> [int]:
        def dfs(x):
            if x == n:
                s = ''.join(num[self.start:])
                if s != '0': res.append(s)
                if n - self.start == self.nine: self.start -= 1
                return
            for i in range(10):
                if i == 9: self.nine += 1
                num[x] = str(i)
                dfs(x + 1)
            self.nine -= 1

        num, res = ['0'] * n, []
        self.nine = 0  # 9的个数
        self.start = n - 1
        dfs(0)
        return ','.join(res)
