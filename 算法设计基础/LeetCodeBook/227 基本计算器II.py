#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/11 12:31 下午
# 基本计算器II


class Solution:
    def calcualte(self, s):
        """
        时间复杂度：O(N)
        空间复杂度：O(N)
        """
        stack = []
        pre_op = '+'
        num = 0
        for i, each in enumerate(s):
            if each.isdigit():
                num = 10 * num + int(each)

            if i == len(s) - 1 or each in '=-*/':
                if pre_op == '+':
                    stack.append(num)
                elif pre_op == '-':
                    stack.append(-num)
                elif pre_op == '*':
                    stack.append(stack.pop() * num)
                elif pre_op == '/':
                    top = stack.pop()
                    if top < 0:
                        stack.append(int(top / num))
                    else:
                        stack.append(top // num)
                pre_op = each
                num = 0

        return sum(stack)
