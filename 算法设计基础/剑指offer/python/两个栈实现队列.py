
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-10-07 10:14:53
# 描述: 两个栈实现队列


class CQueue:

    def __init__(self):
        self.A, self.B = [], []

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        if self.B:
            return self.B.pop()

        if not self.A:
            return -1
        while self.A:
            self.B.append(self.A.pop())

        return self.B.pop()
