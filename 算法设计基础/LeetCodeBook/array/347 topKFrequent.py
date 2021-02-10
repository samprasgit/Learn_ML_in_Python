# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
排序算法
最小堆
桶排序
"""

import heapq


class Solution(object):
    def topKFrequent(self, nums, k):
        """
        桶排序
        :param nums: List[int]
        :param k: int
        :return: List[int]
        """
        # 统计元素的频率
        freq_dict = dict()
        for num in nums:
            freq_dict[num] = freq_dict.get(num, 0) + 1
        print(freq_dict)

        # 桶排序
        bucket = [[] for _ in range(len(nums) + 1)]
        for key, value in freq_dict.items():
            bucket[value].append(key)

        print(bucket)

        # 逆序取出前K个元素
        ret = list()

        for i in range(len(nums), -1, -1):
            if bucket[i]:
                ret.extend(bucket[i])
            if len(ret) >= k:
                break
        return ret[:k]

    # def topKFrequent(self, nums, k):
    #
    #     # 统计元素的频率
    #     freq_dict = dict()
    #     for num in nums:
    #         freq_dict[num] = freq_dict.get(num, 0) + 1
    #
    #     # 维护一个大小k为的最小堆，使得堆中的元素即为前K个高频元素
    #     pq = list()
    #     for key, value in freq_dict.items():
    #         if len(pq) < k:
    #             heapq.heappush(pq, (value, key))
    #         elif value > pq[0][0]:
    #             heapq.heapplace(pq, (value, key))
    #
    #     # 取出堆中的元素
    #     ret = list()
    #     while pq:
    #         ret.append(heapq.heappop(pq)[1])
    #     return ret


if __name__ == "__main__":
    s = Solution()
    nums, k = [1, 1, 1, 2, 2, 3], 2
    res = s.topKFrequent(nums, k)
    print(res)
