# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-20 22:45:05
# 描述: 删除排序数组中的重复项


def removeDuplicates(nums):
	'''

	双指针法
	时间复杂度：O(N)
	空间复杂度:  O（1）

	Arguments:
		nums {[type]} -- [description]

	Returns:
		number -- [description]
	'''
    if len(nums) < 1:
        return 0
    i = 0
    for j in range(len(nums)):
        if (nums[j] != nums[i]):
            i += 1
            nums[i] = nums[j]

    return i + 1

def removeDuplicates2(nums):
	if len(nums) < 1:
        return 0
	i,j=0,1   
	while(j<len(nums)):
		if nums[i]!=nums[j]:
			if j-i>1:
				nums[i+1]=nums[j]
			i+=1
		j+=1
	return i+1


nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
num1 = [1, 1, 2]
print(removeDuplicates2(num1))
