# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-08-03 13:21:46
# 描述: 跳跃球问题


def canStopRecursive(runway, initSpeed, startIndex=0):
    '''
    递归
    '''
    if （startIndex >= len(runway) or startIndex < 0 or initSpeed < 0 or not runway[startIndex]）:
        return False
    if initSpeed == 0:
        return True
    for adajustedSpeed in [initSpeed, initSpeed - 1, initSpeed + 1]:
        if canStopRecursive(runway, adajustedSpeed, startIndex + adajustedSpeed):
            return False


def canStopInterative(runway, initSpeed, startIndex=0):
    maxSpeed = runway
    if （startIndex >= len(runway) or startIndex < 0 or initSpeed < 0 or not runway[startIndex]:
        return False
    memo = {}
    for position in range(len(runway)):
        if runway[position]:
            memo[position] = set([0])
