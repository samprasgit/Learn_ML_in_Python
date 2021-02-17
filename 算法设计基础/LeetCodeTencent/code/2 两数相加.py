#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/10 5:36 下午
# Author : samprasgit
# desc :

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1, l2):
        def dfs(l1, l2, i):
            if not l1 and not l2 and not i:
                return None
            s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + i
            node = ListNode(s % 10)
            node.next = dfs(l1.next if l1 else None, l2.next if l2 else None, s // 10)
            return node

        return dfs(l1, l2, 0)
