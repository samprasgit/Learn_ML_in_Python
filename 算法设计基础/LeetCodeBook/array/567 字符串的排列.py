#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/10 12:21 下午
# Author : samprasgit
# desc :

import collections


class Solution:
    def checkInclusion(self, s1, s2):
        """
        滑动窗口 + 字典
        """
        # 统计s1中出现的次数
        counter1 = collections.Counter(s1)
        n = len(s2)
        # 定义滑动窗口范围   「left，right」闭区间 长度与s1相等
        left = 0
        right = len(s1) - 1
        # 统计窗口s2[left,right-1]内出现的元素次数
        counter2 = collections.Counter(s2[:right])
        while right < n:
            # 把right位置元素放到counter2中
            counter2[s2[right]] += 1
            # 如果滑动窗口内各个元素出现的次数跟 s1 的元素出现次数完全一致，返回 True
            if counter1 == counter2:
                return True
            # 窗口向右移动前，把当前 left 位置的元素出现次数 - 1
            counter2[s2[left]] -= 1
            # 如果当前 left 位置的元素出现次数为 0， 需要从字典中删除，否则这个出现次数为 0 的元素会影响两 counter 之间的比较
            if counter2[s2[left]] == 0:
                del counter2[s2[left]]
            # 窗口向右移动
            left += 1
            right += 1

        return False
