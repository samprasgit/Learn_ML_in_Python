#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/18 2:56 下午
def isPalindrome(x):
    """
    双指针
    """
    lst = list(str(x))
    L, R = 0, len(lst) - 1
    while L <= R:
        if lst[L] != lst[R]:
            return False
        L += 1
        R -= 1
    return True
x=10
print(isPalindrome(x))