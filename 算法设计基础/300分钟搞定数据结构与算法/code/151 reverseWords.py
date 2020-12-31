# -*- coding: utf-8 -*-


def reverseWords(s):
    return ' '.join(s.split()[::-1])


print(reverseWords("the sky is the blue !"))
