# ! /usr/bin/env python
# -*- coding: utf-8 -*-


import math


def twoSum(numbers, target):
    # 定义low，high指针分别处在数组两端
    low, high = 0, len(numbers) - 1
    while low < high:
        sums = numbers[low] + numbers[high]
        # 如果两指针之和为target值，返回索引+1
        if sums == target:
            return low + 1, high + 1
        # 如果两指针指向值之和大于target值，high指针左移
        elif sums > target:
            high -= 1
        # 如果两指针指向值之和小于target值，low指针右移
        else:
            low += 1


def judgeSquareSum(target):
    low, high = 0, int(math.sqrt(target))
    while low <= high:  # 注意等号 没有等号 2 返回错误
        sums = low * low + high * high
        if sums == target:
            return True
        elif sums > target:
            high -= 1
        else:
            low += 1

    return False


def reverseVowels(s):
    dic = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
    left, right = 0, len(s) - 1
    l = list(s)  # str类型数据无法直接查询in和not in，转换为list
    while left < right:
        if l[left] in dic and l[right] in dic:
            l[left], l[right] = l[right], l[left]
            left += 1
            right -= 1
        if l[left] not in dic:
            left += 1
        if l[right] not in dic:
            right -= 1

    return ''.join(l)


def validPalindrome(s):
    left, right = 0, len(s) - 1
    if s == s[::-1]:  # 本身是回文串
        return True
    else:
        for n in range(len(s)):
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                # 先删除左边
                s_L = s[:left] + s[left + 1:]
                if s_L == s_L[::-1]:
                    return True

                s_R = s[:right] + s[right + 1:]
                if s_R == s_R[::-1]:
                    return True

                return False


def merge1(nums1, m, nums2, n):
    # 从后往前
    # copy nums1
    nums1_copy = nums1[:m]
    nums1[:] = []
    # 两个指针
    p1, p2 = 0, 0
    # 比较
    while p1 < m and p2 < n:
        if nums1_copy[p1] < nums2[p2]:
            nums1.append(nums1_copy[p1])
            p1 += 1

        else:
            nums1.append(nums2[p2])
            p2 += 1
    # 加上剩下的
    if p1 < m:
        nums1[p1 + p2:] = nums1_copy[p1:]
    if p2 < n:
        nums1[p1 + p2:] = nums2[p2:]

    return nums1


class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None

# def merge2(nums1, m, nums2, n):


# def hasCycle(head):
#     if not head:
#         return head
#     slow = head
#     quick = head
#     while quick and slow:
#         slow = slow.next
#         # 判断quick quick.next是否都为空
#         if quick.next:
#             quick = quick.next.next
#         else:
#             return False
#         if quick is slow:
#             return True

#     return False


def isSubStr(s, dic):
    i = 0
    for di in dic:
        if di == s[i]:
            i += 1
        if i == len(s):
            return True
    return False


def findLongestWord(s, d):
    # 按大到小和字典序排列
    d.sort(key=lambda x: [-len(x), x])
    # 初始化最大匹配
    longestWord = ""
    for word in d:
        if isSubStr(word, s):
            if len(word) > len(longestWord):
                # 找到更大的匹配
                longestWord = word

    return longestWord


numbers = [2, 7, 11, 15]
target = 2
s = ' hello'
s1 = 'acab'
nums1, m = [1, 2, 3, 0, 0, 0],  3
nums2, n = [2, 5, 6],        3
head, pos = [3, 2, 0, -4], 1
s, d = "abpcplea",  ["a", "p", "m", "e"]


print(reverseVowels(s))
print(validPalindrome(s1))
print(merge1(nums1, m, nums2, n))
# print("是否存在循环列表:", hasCycle(head))
print(findLongestWord(s, d))
