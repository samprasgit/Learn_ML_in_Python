# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   1503 所有蚂蚁掉下来前的最后一刻.py
@Time    :   2021/05/09 15:24:40
@Desc    :   None
"""


class Solution:
    def getLastMoment(self, n, left, right):
        """
        两只蚂蚁相撞后同时改变移动方向相当于是 两只蚂蚁都不改变方向
        一只蚂蚁的位置是p   max(p,n-p)
        """
        lastMoment = 0 if not left else max(left)
        if right:
            lastMoment = max(lastMoment, max(n-ant for ant in right))
        return lastMoment
