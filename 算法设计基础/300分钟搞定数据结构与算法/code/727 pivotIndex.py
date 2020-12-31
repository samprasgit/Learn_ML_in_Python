# !/usr/bin/env python

"""
给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。

我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。

如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

"""


def pivotIndex1(nums):
	'''
	暴力破解法
	'''
    if nums == None:
        return -1
    for i in range(len(nums)):
        left, right = 0, 0
        for j in range(i):
            left += int(nums[j])
        for k in range(i + 1, len(nums)):
            right += int(nums[k])

        if left == right:
            return i
        return -1


def pivotIndex2(nums):
    llast, rlast = 0, sum(nums[1:])
    for i in range(len(nums)):
        if llast == rlast:
            return i
        if i == len(nums) - 1:
            return -1
        llast += nums[i]
        rlast -= nums[i + 1]
    return -1


def pivotIndex3(nums):
	"""
	进一步优化
	"""
    left = 0
    total = sum(nums)
    for i, j in enumerate(nums):
        if left == (total - j) / 2:
            return i
        left += j
    return -1


nums = [1, 7, 3, 6, 5, 6]
print(pivotIndex3(nums))
