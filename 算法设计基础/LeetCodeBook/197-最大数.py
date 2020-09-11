# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-09-11 09:33:49
# 描述: 构建最大数字


class LargerNumKey(str):

    def __lt__(x, y):
        return x + y > y + x


class Solution:

    def largestNumber(self, nums):
        largest_num = ''.join(sorted(map(str, nums)), key=LargerNumKey)
        return '0' if largest_num[0] == '0' else largest_num
