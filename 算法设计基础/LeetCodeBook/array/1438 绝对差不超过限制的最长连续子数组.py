#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/21 8:49 下午
class Solution:
    def longestSubarray(self, nums, limit):
        """
        滑动窗口
        sortedcontainers
        """
        from sortedcontainers import SortedList
        s = SortedList()
        left, right = 0, 0
        res = 0
        while right < len(nums):
            s.add(nums[right])
            while s[-1] - s[0] > limit:
                s.remove(nums[left])
                left += 1

            res = max(res, right - left + 1)
            right += 1

        return res


if __name__ == "__main__":
    S = Solution()
    nums = [8, 2, 4, 7]
    limit = 4
    print(S.longestSubarray(nums, limit))
