#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/22 7:52 下午
# 设计一个停发系统

class ParkingSystem:
    """
    时间复杂度：O(1)
    空间复杂度：O(1)
    """
    def __init__(self, big, medium, small):
        self.park = [0, big, medium, small]

    def addCar(self, carType):
        if self.park[carType] == 0:
            return False
        self.park[carType] -= 1
        return True
