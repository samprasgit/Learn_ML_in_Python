# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-29 12:26:52
# 描述: 四数之和
# 难点 四元组不重复


class Solution:

    def fourSum(self, nums, target):
        '''
        排序+双指针

        '''
        n = len(nums)
        res = []
        if not nums or n < 4:
            return res
        nums.sort()

        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(n - 2):
                if j - i > 1 and nums[j] == nums[j - 1]:
                    continue
                # 第一个指针
                left = j + 1
                right = n - 1

                while left < right:
                    cur = nums[i] + nums[j] + nums[left] + nums[right]
                    if cur == target:
                        if res not in res:
                            res.append(
                                [nums[i], nums[j], nums[left], nums[right]])

                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif cur < target:
                        left += 1
                    else:
                        right -= 1
        return res


if __name__ == "__main__":
    s = Solution()
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    res = s.fourSum(nums, target)
    print(res)
