#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/11 2:29 下午
# Author : samprasgit
# desc :
# TopK算法https://mp.weixin.qq.com/s/FFsvWXiaZK96PtUg-mmtEw
# 为什么使用小根堆？
# 因为我们需要在堆中保留数据流中的前 KK 大元素，使用小根堆能保证每次调用堆的 pop() 函数时，从堆中删除的是堆中的最小的元素（堆顶）。
# 为什么能保证堆顶元素是第 KK 大元素？
# 因为小根堆中保留的一直是堆中的前 KK 大的元素，堆的大小是 KK，所以堆顶元素是第 KK 大元素。
# 每次 add() 的时间复杂度是多少？
# 每次 add() 时，调用了堆的 push() 和 pop() 方法，两个操作的时间复杂度都是 log(K)log(K).
import heapq


class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.que = nums
        heapq.heapify(self.que)

    def add(self, val):
        heapq.heappush(self.que, val)
        while len(self.que) > self.k:
            heapq.heappop(self.que)

        return self.que[0]
