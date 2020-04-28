# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-28 22:08:34
# 描述: 三数之和
# 难点：去重


class Solution1:

    # hash-map
    # 超过了时间限制

    def threeSum(self, nums):
        target = 0
        dic = {}
        l = []
        for i, num in enumerate(nums):
            dic[num] = i
        for i in range(len(nums)):
            for j in range(len(nums)):
                c = target - (nums[i] + nums[j])
                if (i != j) and c in nums and (i != dic[c]) and (j != dic[c]):
                    l.append(sorted([nums[i], nums[j], c]))
        return ([list(t) for t in (set([tuple(t) for t in l]))])


class Solution2:
    # 双指针

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
    s = Solution2()

    nums = [-1, 0, 1, 2, -1, -4]
    print(s.threeSum(nums))
