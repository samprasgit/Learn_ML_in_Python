# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-20 22:04:01
# 描述: 盛水最多的容器


class Solution1:
    '''

    双指针:
    面积由水槽的短板决定
    最大面积S=min(height(i),height(j))*(j-i)
    合理性证明？
    时间复杂度：O(N)
    空间复杂度：O(1)

    '''

    def maxArea(self, height):
        i, j, res = 0, len(height) - 1, 0
        while i < j:
            if height[i] < height[j]:
                res = max(res, height[i] * (j - i))
                i += 1
            else:
                res = max(res, height[j] * (j - i))
                j -= 1
        return res


nums = [1, 8, 6, 2, 5, 4, 8, 3, 7]


s = Solution1()
print(s.maxArea(nums))
