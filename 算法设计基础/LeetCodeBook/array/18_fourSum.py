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
            # 去重
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            # 如果固定数与数组三最小数之和大于target,则后续循环都是不存在解的，从遍历中跳出
            if nums[i] + sum(nums[i + 1:i + 3 + 1]) > target:
                break
            # 如果固定数与数组三最大数之和小于target, 则当前遍历不存在解, 进入下一个遍历’
            if nums[i] + sum(nums[-1:-3 - 1:-1]) < target:
                continue
            for j in range(i + 1, n - 2):
                # 去重
                if j - i > 1 and nums[j] == nums[j - 1]:
                    continue
                if nums[i] + nums[j] + sum(nums[j + 1:j + 2 + 1]) > target:
                    break
                if nums[i] + nums[j] + sum(nums[-1:-2 - 1:-1]) < target:
                    continue
                left = j + 1

                right = n - 1

                while left < right:
                    cur = nums[i] + nums[j] + nums[left] + nums[right]
                    if cur == target:

                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        # 求得正解之后去重解
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
