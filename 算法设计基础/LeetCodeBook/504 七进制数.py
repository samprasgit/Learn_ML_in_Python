#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/26 5:41 下午
# 进制转换

class Solution:
    def convertToBase7(self, num):
        """
        递归
        """
        if num < 0:
            return "-" + self.convertToBase7(-num)
        if num < 7:
            return str(num)
        return self.convertToBase7(num // 7) + str(num % 7)

    def solve(self, M, N):
        """
        M 输入十进制数
        N 待转换到的进制
        res 字符串 逆序输出
        时间复杂度：O()
        """
        t = '0123456789ABCDEF'
        res = ""
        if M == 0:
            return 0
        flag = False

        if M < 0:
            flag = True
            M = -M
        while M:
            res += t[M % N]
            M //= N
        if flag:
            return "-" + res[::-1]
        else:
            return res[::-1]
