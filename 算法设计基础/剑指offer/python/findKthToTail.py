# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-21 09:58:33
# 描述: 链表中倒数第k个结点


class ListNode():

    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    def findKthToTail(self, head, k):
        '''
        栈先进先出		
        '''
    node = None
    stack = []
    temp = head
    while temp:
        stack.append(temp)
        temp = temp.next

    if len(stack) >= k:
        for i in range(k):
            node = stack.pop()
        return node

    else:
        return None

    def findKthToTail2(self, head, k):
        '''
        快慢指针：
        构建两个相距为k的指针，当快指针走到链表最后的时候，慢指针就指向倒数第k个节点

        '''
        fast = slow = head
        # 快指针先走K步
        for i in range(k):
            if fast:
                fast = fast.next
            else:  # 链表的长度小于K时，应该返回None
                return None
        # 快指针走了k步之后，快慢指针一块走，直到fast走到链表尾部
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
