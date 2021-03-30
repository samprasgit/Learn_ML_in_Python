#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/27 9:07 上午
# 输入一个以#结束的字符串，本题要求滤去所有的非十六进制字符（不分大小写），组成一个新的表示十六进制数字的字符串，然后将其转换为十进制数后输出。如果在第一个十六进制字符之前存在字符“-”，则代表该数是负数。
# 输入格式 ±P-xf4±1!#
# 输出格式 -3905


# s = input().strip('#')
s = '±P-xf4±1!#'


def convertToBase(s):
    """
    十六进制转换到十进制
    """
    s = s.upper()
    flag = 0
    str = ""
    m = 0
    for i in s:
        if i == '-' and flag == 0:
            m = 1
        if (i >= '0' and i <= '9') or (i >= 'A' and i <= 'F'):
            flag = 1
            str += i
    if str == "":
        print(0)
    elif m == 1:
        print(-int(str, 16))
    else:
        print(int(str, 16))
    # int(x,base)


def convertToBase2(s):
    s = s.upper()
    str = ""
    flag = 0
    m = 0
    for i in s:
        if i == '-' and flag == 0:
            m = 1
        if (i >= '0' and i <= '9') and (i >= 'A' and i <= 'F'):
            flag = 1
            str += i

    res = 0
    q = 1
    for i in range(len(str) - 1, -1, -1):
        res += dict[str[i]] * q
        q *= 16

    if m == 1:
        res = -res
    print(int(res))
