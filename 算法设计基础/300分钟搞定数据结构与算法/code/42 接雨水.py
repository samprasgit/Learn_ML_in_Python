# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-10 09:19:26
# 描述: 接雨水

class Solution1:
    def trap(self, height):
        """
        """
        n = len(height)
        s1, s2 = 0, 0
        max1, max2 = 0, 0

        for i in range(n):
            if height[i] > max1:
                max1 = height[i]
            if height[n - i - 1] > max2:
                max2 = height[n - i - 1]
            s1 += max1
            s2 += max2

        res = s1 + s2 - max1 * len(height) - sum(height)

        return res


if __name__ == "__main__":
    s = Solution1()
    hei = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(s.trap(hei))
