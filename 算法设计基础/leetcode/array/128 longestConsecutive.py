#! /usr/bin/env python
# -*- coding: utf-8 -*-


# def longestConsecutive(nums):
#     longest_streak = 0
#
#     for num in nums:
#         current_num = num
#         current_streak = 1
#
#         while current_num + 1 in nums:
#             current_num += 1
#             current_streak += 1
#
#         longest_streak = max(longest_streak, current_streak)
#
#     return longest_streak

# def longestConsecutive(nums):
#     '''
#     排序
#     时间复杂度O(nlog(n))
#     :param nums:
#     :return:
#     '''
#
#     if not nums:
#         return 0
#     nums.sort()
#
#     longest_streak = 1
#     current_streak = 1
#
#     for i in range(1, len(nums)):
#         if nums[i] != nums[i - 1]:
#             if nums[i] == nums[i - 1] + 1:
#                 current_streak += 1
#             else:
#                 longest_streak = max(longest_streak, current_streak)
#                 current_streak = 1
#
#     return max(longest_streak, current_streak)


def longestConsecutive(nums):
    '''
    哈希表和线性空间
    :param nums:
    :return:
    '''
    longest_streak = 0
    num_set = set(nums)
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            # longest_streak = max(longest_streak, current_streak)
            if current_streak > longest_streak:
                longest_streak = current_streak

    return longest_streak





nums = [100, 4, 200, 1, 3, 2]
nums2 = [9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6]
print(longestConsecutive(nums2))
