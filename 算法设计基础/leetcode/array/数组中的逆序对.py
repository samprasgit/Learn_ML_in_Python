# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-24 23:56:46
# 描述: 数组中的逆序对
from typing import List


class Solution1:
    '''

    暴力解法
    时间复杂度：O(N^2)
    空间复杂度:O(1)

    '''

    def reversePairs(self, nums):
        n = len(nums)
        if n < 2:
            return 0

        res = 0
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                if nums[i] > nums[j]:
                    res += 1

        return res


class Solution2:
    """

    分治思想
    使用归并排序统计逆序数：
    - 左边的逆序对
    - 右边的逆序对
    - 横跨两个区的逆序对

    """

    def reversePairs(self, nums):
        n = len(nums)
        if n < 2:
            return 0
        # 用于递归的辅助数组
        temp = [0 for _ in range(n)]
        return self.count_reverse_pairs(nums, 0, n - 1, temp)

    def count_reverse_pairs(self, nums, left, right, temp):
        # 在数组nums的区间[left,right]统计逆序对
        if left == right:
            return 0
        mid = (left + right) // 2
        left_pairs = self.count_reverse_pairs(nums, left, mid, temp)
        right_pairs = self.count_reverse_pairs(nums, mid + 1, right, temp)
        reverse_pairs = left_pairs + right_pairs
        if nums[mid] <= nums[mid + 1]:
            return reverse_pairs
        reverse_cross_pairs = self.merge_and_count(
            nums, left, mid, right, temp)
        return reverse_pairs + reverse_cross_pairs

    def merge_and_count(self, nums, left, mid, right, temp):
        for i in range(left, right + 1):
            temp[i] = nums[i]

        i = left
        j = mid + 1
        res = 0
        for k in range(left, right + 1):
            if i > mid:
                nums[k] = temp[j]
                j += 1

            elif j > right:
                nums[k] = temp[i]
                i += 1
            elif temp[i] <= temp[j]:
                nums[k] = temp[i]
                i += 1
            else:
                nums[k] = temp[j]
                j += 1
                res += (mid - i + 1)
        return res


if __name__ == "__main__":
    s2 = Solution2()
    nums = [7, 5, 6, 4]
    print(s2.reversePairs(nums))
