# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-28 22:08:34
# 描述: 三数之和
# 难点：去重


class Solution1:

    '''hash-map
    时间复杂度：
    空间复杂度：
    '''

    def threeSum(self, nums):
        '''
        Arguments:
            nums {[int]} -- [description]
        '''
        if len(nums) < 3:
            return []
        '''先对数组排序, 遍历数组遇到与前一个元素相同的情况可直接跳过'''
        nums.sort()
        target_hash = {-x: i for i, x in enumerate(nums)}
        res = []
        res_hash = {}
        for i, first in enumerate(nums):
            '''当前元素与前一个元素相同时, 可直接跳过以优化性能'''
            if i > 0 and first == nums[i - 1]:
                continue
            for j, second in enumerate(nums[i + 1:]):
                '''检查两数之和是否存在于哈希表中'''
                if first + second in target_hash:
                    target_index = target_hash[first + second]
                    if target_index == i or target_index == i + j + 1:
                        continue
                    '''将找到的结果存入另一个哈希表中, 避免包含重复结果'''
                    row = sorted([first, second, nums[target_index]])
                    key = ",".join([str(x) for x in row])
                    if key not in res_hash:
                        res.append(row)
                        res_hash[key] = True
        return res


class Solution2:
   	'''
    三指针
    '''


    def threeSum(self, nums):
        n = len(nums)
        res = []
        if (not nums or n < 3):
            return []
        nums.sort()
        for i in range(n):
            if (nums[i] > 0):
                return res
            if (i > 0 and nums[i] == nums[i - 1]):
                continue
            L = i + 1
            R = n - 1
            while (L < R):
                if (nums[i] + nums[L] + nums[R] == 0):
                    res.append([nums[i], nums[L], nums[R]])
                    while (L < R and nums[L] == nums[L + 1]):
                        L += 1
                    while (L < R and nums[R] == nums[R - 1]):
                        R -= 1
                    L += 1
                    R -= 1
                elif(nums[i] + nums[L] + nums[R] > 0):
                    R -= 1
                else:
                    L += 1

        return res


if __name__ == "__main__":
    s = Solution1()

    nums = [-1, 0, 1, 2, -1, -4]
    print(s.threeSum(nums))
