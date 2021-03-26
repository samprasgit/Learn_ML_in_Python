#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/25 8:13 上午
# 直接遍历链表

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def deleteDuplicates(self, head):
        """
        一次遍历
        时间复杂度：O(N)
        空间复杂度：O(N)
        """
        if not head:
            return head
        # 头结点 可能也是重复值会被删除
        dummy = ListNode(0, head)
        cur = dummy

        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                x = cur.next.val
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next

        return dummy.next
