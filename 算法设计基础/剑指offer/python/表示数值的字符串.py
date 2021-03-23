#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/23 9:41 上午
# 有限状态自动机

class Solution:
    def isNumber(self, s):
        # 0. start with 'blank'
        # 1. 'sign' before 'e'
        # 2. 'digit' before 'dot'
        # 3. 'digit' after 'dot'
        # 4. 'digit' after 'dot' (‘blank’ before 'dot')
        # 5. 'e'
        # 6. 'sign' after 'e'
        # 7. 'digit' after 'e'
        # 8. end with 'blank'

        states = [
            {' ': 0, 's': 1, 'd': 2, '.': 4},
            {'d': 2, '.': 4},
            {'d': 2, '.': 3, 'e': 5, ' ': 8},
            {'d': 3, 'e': 5, ' ': 8},
            {'d': 3},
            {'s': 6, 'd': 7},
            {'d': 7},
            {'d': 7, ' ': 8},
            {' ': 8}

        ]
        p = 0
        for c in s:
            # digit
            if '0' <= c <= '9':
                t = 'd'
            # sign
            elif c in "+-":
                t = 's'
            # e E
            elif c in "eE":
                t = 'e'
            # dot blank
            elif c in ". ":
                t = c
            else:
                t = '?'

            if t not in states[p]:
                return False

            p = states[p][t]

        return p in (2, 3, 7, 8)



    def isNumber2(self,s):
        """
        float 强转 如果可以是一个合法的数值字符串
        """
        try:
            float(s)
        except:
            return 0

