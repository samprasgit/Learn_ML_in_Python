# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-07 11:58:20
# 描述: 二分查找以及变形


# https://www.cnblogs.com/gongpixin/p/6761389.html


def binary_chop(nums, target):
    '''

    非递归 二分查找

    最优时间复杂度  O(1)
    最坏时间复杂度 O(logn)

    Arguments:
            list {[type]} -- [description]
            target {[type]} -- [description]
    '''
    n = len(nums)
    left = 0
    right = n - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            return True
    return False


def binary_chop2(nums, target):
    '''
    递归  二分查找

    [description]

    Arguments:
            nums {[type]} -- [description]
            target {[type]} -- [description]
    '''
    n = len(nums)
    if n < 1:
        return False
    mid = n // 2
    if nums[mid] > target:
        return bninary(nums[0:mid], target)
    elif nums[mid] < target:
        return bnbinary(nums[mid + 1:], target)
    else:
        return True
#  查找最后一个小于key的元素


def findLastKey(nums, key):
    n = len(nums)
    left = 0
    right = n - 1
    while left <= right:
        mid = (left + right) // 2
        if key <= nums[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return right

# 查找第第一个大于key的元素


def findFirstGreaterKey(nums, keys):
    n = len(nums)
    left = 0
    right = n - 1
    while left <= right:
        mid = (left + right) // 2
        if key <= nums[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return left

# 查找最后一个小于等于Key的袁元素


# 查找第一个大于key的元素
# 查找第一个与key相等的元素
# 查找最后一个与key相等的元素

lists = [2, 4, 5, 12, 14, 23]
data = 12
if binary_chop2(lists, data):
    print("ok")
else:
    print("false")


def findIP(dics, target):
    for nums, item in enumerate(dics):
        if dics[item][0][0][0] == target[0]:
            return item


dict = {'北京':  [("1.1.1.0", "1.1.1.23"), ("1.1.2.0", "1.1.3.123")], '广东': [
    ("2.1.1.0", "2.1.1.23"), ("2.1.2.0", "2.1.3.123")]}
target = ('1.1.2.128')
print(findIP(dict, target))
