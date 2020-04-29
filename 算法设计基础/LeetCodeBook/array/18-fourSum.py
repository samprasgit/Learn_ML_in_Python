# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-29 16:00:43
# 描述: 四数之和
from typing import List


class Solution(object):

    def fourSum(self, nums, target):
        """
        排序+双指针
        去重
        :param nums:
        :param target:
        :return:
        """
        n = len(nums)
        nums.sort()
        res = []
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                left = j + 1  # 第一个指针
                right = n - 1  # 第二个指针
                current = nums[i] + nums[j]
                while (left < right):
                    if (current + nums[left] + nums[right] == target):
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        # 第一个指针j去重
                        while (left < right and nums[left] == nums[left + 1]):
                            left += 1
                        # 第二个指针去重
                        while (left < right and nums[right] == nums[right - 1]):
                            right -= 1

                    elif (current + nums[left] + nums[right] < target):
                        left += 1
                    else:
                        right -= 1
                while (j < n - 2 and nums[j] == nums[j - 1]):
                    j += 1
            while (i < n - 3 and nums[i] == nums[i - 1]):
                i += 1

        return res


if __name__ == "__main__":
    s = Solution()
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    res = s.fourSum(nums, target)
    print(res)
