# !/usr/bin/env python
#


def domainindex():
    maxval, index = 0, -1
    for i, j in enumerate(nums):
        if maxval < j:
            maxval = j
            index = i
    for i, j in enumerate(nums):
        if index == j:
            continue
        if maxval < 2 * j:
            return -1
    return index


def domainIndex(nums):
    if len(nums) == 1:
        return 0
    nums1 = sorted(nums, reverse=True)
    if nums1[0] >= 2 * nums1[1]:
        return nums.index(nums1[0])
    return -1


# 测试实例

nums = [1, 2, 3, 4]

print(domainIndex(nums))
