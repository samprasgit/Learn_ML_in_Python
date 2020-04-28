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
    # 在第 2 个子区间元素归并回去的时候，计算逆序对的个数

    def reversePairs(self, nums):
        n = len(nums)
        if n < 2:
            return 0
        # 用于递归的辅助数组
        temp = [0 for _ in range(n)]
        return self.count_reverse_pairs(nums, 0, n - 1, temp)

    def count_reverse_pairs(self, nums, left, right, temp):
        '''[summary]

        [description]

        Arguments:
                nums {[type]} -- [description]
                left {[type]} -- [description]
                right {[type]} -- [description]
                temp {[type]} -- [description]

        Returns:
                number -- [description]
        '''
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


class FenwickTree:

    def __init__(self, n):
        self.size = n
        self.tree = [0 for _ in range(n + 1)]

    def __lowbit(self, index):
        return index & (-index)

    # 单点更细：从上往下，最多到len,可以取等
    def update(self, index, delta):
        while (index <= self.size):
            self.tree[index] += delta
            index += self.__lowbit(index)

    # 区间查询，从上往下，最少到1，可以取等
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= self.__lowbit(index)

        return res


class Solution3:
    '''
    树状数组
    '''

    def reversePairs(self, nums):
        size = len(nums)
        if size < 2:
            return 0
        # 离散化 原始数组去重从小大大排序
        s = list(set(nums))
        # 构造最小堆，从小到大一个个拿出来
        import heapq
        heapq.heapify(s)
        # 由数字查排名
        rank_map = dict()
        rank = 1
        # 不重复的数字
        rank_map_size = len(s)
        for _ in range(rank_map_size):
            num = heapq.heappop(s)
            rank_map[num] = rank
            rank += 1

        res = 0
        ft = FenwickTree(rank_map_size)
        # 从后往前看，拿出一个数字，就更一下，然后向前查询比它小的个数
        for i in range(size - 1, -1, -1):
            rank = rank_map[nums[i]]
            ft.update(rank, 1)
            res += ft.query(rank - 1)
        return res


if __name__ == "__main__":
    s2 = Solution3()
    nums = [7, 5, 6, 4]
    num2 = [2, 3, 5, 7, 1, 4, 6, 8]
    print(s2.reversePairs(num2))
