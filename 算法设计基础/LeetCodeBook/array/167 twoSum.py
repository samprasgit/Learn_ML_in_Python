# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-26 21:26:07
# 描述: 两数之和 II - 输入有序数组


class Solution1:
    # 双指针

    def twoSum(self, numbers, target):
        l, r = 0, len(numbers) - 1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l + 1, r + 1]
            elif s < target:
                l += 1
            else:
                r -= 1


class Solution2:
    # hash-map
    # 一次hash

    def twoSum(self, numbers, target):
        dic = {}
        for i, num in enumerate(numbers):
            if target - num in dic:
                return [dic[target - num] + 1, i + 1]
            dic[num] = i


class Solution3:
        # binary search

    def twoSum(self, numbers, target):
        for i in range(len(numbers)):
            l, r = i + 1, len(numbers) - 1
            tmp = target - numbers[i]
            while l <= r:
                mid = l + (l - r) // 2
                if numbers[mid] == tmp:
                    return [i + 1, mid + 1]
                elif numbers[mid] < tmp:
                    l = mid + 1
                else:
                    r = mid - 1


if __name__ == "__main__":
    s = Solution3()
    numbers, target = [2, 7, 11, 15], 9
    print(s.twoSum(numbers, target))
