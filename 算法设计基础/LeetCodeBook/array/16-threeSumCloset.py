# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-29 10:31:38
# 描述: 最接近的三数之和


class Solution(object):
    '''
    排序和双指针
    时间复杂度：O(N^2)


    '''

    def threeSumCloset(self, nums, target):
        n = len(nums)
        if n < 3:
            return None

        nums.sort()

        ans = nums[0] + nums[1] + nums[2]
        for i in range(n - 1):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            j = i + 1

            k = n - 1
            while j < k:
                val = nums[i] + nums[j] + nums[k]
                # 计算差值有点浪费时间 优化？
                if abs(val - target) < abs(ans - target):
                    ans = val

                if val > target:
                    k -= 1
                elif val < target:
                    j += 1
                else:
                    return ans
        return ans


if __name__ == "__main__":
    nums = [-1, 2, 1, -4]
    target = 1
    s = Solution()
    res = s.threeSumCloset(nums, target)
    print(res)
