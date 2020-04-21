# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-21 09:25:12
# 描述: 数值的整数次方


class Solution:

    def power(self, base, exponent):
        return pow(base, exponent)

    def power2(self, base, exponent):
        if exponent == 0:
            return 1
        if base == 0:
            return 0

        flag = True
        if exponent < 0:
            exponent = -exponent
            flag = False
        result = 1
        for i in range(exponent):
            result *= base
        return result if flag else 1 / result

    def power3(self, base, exponent):
        if exponent == 0:
            return 1
        if base == 0:
            return 0

        flag = True
        if exponent < 0:
            exponent = -exponent
            flag = False
        result = 1
        while exponent:
            if exponent & 1:
                result *= base
            base *= base
            exponent = exponent >> 1
        return result if flag else 1 / result
