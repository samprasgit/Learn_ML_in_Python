# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-21 09:34:29
# 描述: 调整数组顺序使奇数位于偶数前面


class Solution:

    def reOrderArray(self, array):
        '''
        建立两个数组
        '''
        oddNumber = []
        evenNUmber = []
        for i in array:
            if i % 2 == 1:
                oddNumber.append(i)
            else:
                evenNUmber.append(i)
        return oddNumber + evenNUmber

    def reOrderArray2(self, array):
        '''
        从后往前遍历数组
        '''
        left = 0
        right = len(array) - 1
        while left < len(array):
            if array[right] % 2 == 0:
                right -= 1
                left += 1
            else:
                temp = array[right]
                for i in range(right, 0, -1):
                    array[i] = array[i - 1]

                array[0] = temp
                left += 1
        return array
