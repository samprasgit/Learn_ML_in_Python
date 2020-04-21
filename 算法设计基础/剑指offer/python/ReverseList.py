# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-21 10:28:44
# 描述: 反转链表


class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    def ReverseList(self, pHead):
        '''
        递归


        '''
        # 链表为空
        if pHead == None:
            return None
        # 链表只有一个节点
        if pHead.next == None:
            return pHead

        temp = pHead.next
        # 对pHead以后的节点的链表反转，res表示反转之后头节点
        res = self.ReverseList(pHead.next)
        # 头结点的next设置为None
        pHead.next = None
        # 头结点的下一节点next设置为pHead,表示反转
        temp.next = pHead
        # 返回头部节点
        return res

    def ReverseList2(self, pHead):
        '''
        利用循环，对每个节点进行反转
        '''
        if (pHead == None or pHead.next == None):
            return pHead

        # 构建一个newHead，并不断更新newHead,直到遍历完链表，这时候的链表就是反转之后的链表

        pre = None
        cur = pHead
        while cur:
                # 记录当前节点的下一节点
            temp = cur.next
            # 当前节点指向pre
            cur.next = pre
            # pre cur 前进一个节点
            pre = cur
            cur = temp
        return pre

    def ReverseList3(self, pHead):
        if (pHead == None or pHead.next == None):
            return pHead

        cur = pHead
        while pHead.next:
            temp = pHead.next.next
            pHead.next.next = cur
            cur = pHead.next
            pHead.next = temp
        return cur
