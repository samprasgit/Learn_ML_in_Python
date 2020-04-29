# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-29 12:26:52
# 描述: 四数之和
# 难点 去除重复解


class Solution:

    def fourSum(self, nums, target):
        n = len(nums)
        nums.sort()
        res = []
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(n - 2):
                if j > 0 and nums[j] == nums[j - 1]:
                    continue
                # 第一个指针
                left = j + 1
                right = n - 1
                cur = nums[i] + nums[j]
                while left < right:
                    if (cur + nums[left] + nums[right] == target):
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        if left > 0 and nums[left] == nums[left - 1]:
                            continue
                        if right > 0 and nums[right] == nums[right + 1]:
                            continue
                    elif cur + nums[left] + nums[right] < target:
                        left += 1
                    else:
                        right += 1
        # set 去除重复解
        res_set = set()
        for item in res:
            res_set.add(item)
        return res_set


if __name__ == "__main__":
    s = Solution()
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    res = s.fourSum(nums, target)
    print(res)
