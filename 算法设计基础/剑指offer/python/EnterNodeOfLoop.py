# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-05-20 11:47:21
# 描述: 链表中环的入口结点


class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    def EnterNodeOfLoop(self, pHead):
    '''

	双指针

	

	Variables:
		if pHead {[type]} -- [description]
		p1 {[type]} -- [description]
		p2 {[type]} -- [description]
		
	'''
        if pHead == None or pHead.next == None:
            return None
        p1 = pHead
        p2 = pHead
        while p1 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                p1 = pHead
                while p1 != p2:
                    p1 = p1.next
                    p2 = p2.next
                return p1
        return None
