# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-09 22:42:14
# 描述: leetcode 557. 反转字符串中的单词 III


def reverseWords(s):
    li = s.split()
    return ' '.join(map(lambda x: x[::-1], li))


s = "Let's take LeetCode contest"
print(reverseWords(s))
