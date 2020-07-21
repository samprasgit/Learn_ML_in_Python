# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-19 23:00:56
# 描述: 只出现一次的数字

from functools import reduce


def singleNumber(nums):
    return reduce(lambda x, y: x ^ y, nums)


print(singleNumber([4, 1, 2, 1, 2]))
