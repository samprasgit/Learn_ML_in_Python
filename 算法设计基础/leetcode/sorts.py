# ! /usr/bin/env python
# -*- coding: utf-8 -*-
from random import randint


class Solution:

    def findKthLargest(self, nums, k):
        if len(nums) < k:
            return []
        index = randint(0, len(nums) - 1)
        pivot = nums[index]
        less = [i for i in nums[:index] + nums[index + 1:] if i < pivot]
        great = [i for i in nums[:index] + nums[index + 1:] if i > pivot]

        if len(great) == k - 1:
            return pivot

        elif len(great) > k - 1:
            return self.findKthLargest(great, k)
        else:
            return self.findKthLargest(less, k - len(great) - 1)
