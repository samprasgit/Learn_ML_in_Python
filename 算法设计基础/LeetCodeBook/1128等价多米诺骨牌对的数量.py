#!/usr/bin/env python
# -*- encoding: utf-8 -*-
class Solution1:
    def numEquivDominoPairs(self, dominoes):
        # 排序 + 哈希表
        dic = dict()

        res = 0
        for d1, d2 in dominoes:
            index = tuple(sorted((d1, d2)))

            if index in dic:
                dic[index] += 1
            else:
                dic[index] = 1

                # 统计数目
        for i in dic:
            res += dic[i] * (dic[i] - 1) // 2
        return res


if __name__ == "__main__":
    s = Solution1()
    dominoes = [[1, 2], [2, 1], [3, 4], [5, 6]]
    print(s.numEquivDominoPairs(dominoes))
