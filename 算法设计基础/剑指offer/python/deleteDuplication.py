# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-05-20 14:34:16
# 描述: 删除链表中重复的节点


class ListNode:

    def __init__(self, x):
        self.vasl = x
        self.next = None


class Solution1:

    def deleteDuplicates(self, pHead):
        '''[summary]

        递归

        Arguments:
                pHead {[type]} -- [description]
        '''
        if pHead == None:
            return None
        if pHead.next == None:
            return pHead
        if pHead.val != pHead.next.val:
            pHead.next = self.deleteDuplicates(pHead.next)
            # 后面的节点递归结束后，返回pHead
            return pHead

        else:
            tempNode = pHead
            while tempNode and tempNode.val == pHead.val:
                tempNode = tempNode.next
            # 重复节点都不留，不保留pHead,直接返回下一个不同节点的递归节点
            return self.deleteDuplicates(tempNode)


class Solution2:

    def deleteDuplicates(self, pHead):
        '''[summary]

        循环

        Arguments:
                pHead {[type]} -- [description]
        '''
