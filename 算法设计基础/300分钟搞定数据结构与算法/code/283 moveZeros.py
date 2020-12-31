# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-09 23:02:50
# 描述: 移动零


def moveZeros(nums):
    """
    快慢指针
    """
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

    return nums


def MoveZeros(nums):
    # 在循环中添加删除
    for num in nums:
        if num == 0:
            nums.remove(0)
            nums.append(0)

    return nums


print(MoveZeros([0, 1, 0, 2, 12]))
