#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Time : 2021/2/5 10:51 上午
# Author : samprasgit
# desc : 尽可能使字符相等

class Solution1:
    def equalSubstring(self, s, t, maxCost):
        #  双指针
        n = len(s)
        costs = [0] * n
        for i in range(n):
            costs[i] = abs(ord(s[i]) - ord(t[i]))
        right, left = 0, 0
        res = 0
        sums = 0
        for right in range(n):
            sums = costs[i]

            if sums > maxCost:
                sums -= costs[i]
                left += 1
            res = max(res, right - left + 1)

        return res


if __name__ == '__main__':
    s = "abcd"
    t = "bcdf"
    maxCost = 3
    solution = Solution1()
    print(solution.equalSubstring(s, t, maxCost))
