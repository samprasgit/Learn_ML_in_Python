# !/usr/bin/env python
# 种花问题


class Solution:
    def canPlaceFlowers(self, flowerbed, n):
        """
        贪心遍历：
        Arguments:
                flowerbed {[type]} -- [description]
        """
        for i in range(len(flowerbed)):
            if (flowerbed[i] == 0) and (i == 0 or flowerbed[i-1] == 0) and(i == len(flowerbed) or flowerbed[i+1]==0):
                n -= 1
                if n <= 0:
                    return True
                flowerbed[i] = 1

        return n <= 0


if __name__ == "__main__":
    s = Solution()
    flowerbed, n = [1, 0, 0, 0, 1], 2
    print(s.canPlaceFlowers(flowerbed, n))
