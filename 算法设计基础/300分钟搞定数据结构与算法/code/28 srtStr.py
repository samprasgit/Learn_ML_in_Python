# !/usr/bin/env python
# -*- coding: utf-8 -*-


def strStr(haystack, needle):
    return haystack.find(needle)


def strStr(haystack, needle):
    # 遍历
    len1 = len(haystack)
    len2 = len(needle)

    if len1 < len2:
        return -1
    start, end = 0, len2 - 1
    while end < len1:
        substr = haystack[start:end + 1]
        if substr == needle:
            return start
        start += 1
        end += 1
    return -1

haystack, needle = "hello", "ll"
print(strStr(haystack, needle))
