# -*- coding:utf-8 -*-
# 合并两个排序的链表
#


class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 返回合并后列表

    def Merge(self, pHead1, pHead2):
        # write code here
        '''
        递归
        时间复杂度 O（m+n）
        思路：
        如果一个链表为空，返回另一个链表
        如果L1节点的值比起L2小，下一个节点应该是l1,返回l1,在return之前，指定l1的下一个节点是L1.next和l2链表的合并后的节点
        '''
        if pHead1 == None and pHead2 == None:
            return None
        if pHead1 == None:
            return pHead2
        if pHead2 == None:
            return pHead1
        if pHead1.val < pHead2.val:
            pHead1.next = self.Merge(pHead1.next, pHead2)
            return pHead1
        else:
            pHead2.next = self.Merge(pHead1, pHead2.next)
            return pHead2

    def Merge2(self, pHead1, pHead2):
        '''
        非递归:迭代
        时间复杂度O(m+n)
        思路：
        构建一个新的链表，然后挨个比较pHead1和pHead2的节点，直到有一个链表遍历完为止
        将没有遍历完的链表直接加在新的链表后面
        '''
        if pHead1 == None and pHead2 == None:
            return None
        if pHead1 == None:
            return pHead2
        if pHead2 == None:
            return pHead1

        # 构建一个新的链表
        newhead = ListNode(-1)
        temp = newhead
        p1, p2 = pHead1, pHead2
        # 有一个为空就停止循环
        while p1 and p2:
            if p1.val < p2.val:
                temp.next = ListNode(p1.val)
                temp = temp.next
                p1 = p1.next

            else:
                temp.next = ListNode(p2.val)
                temp = temp.next
                p2 = p2.next

        if p1:
            temp.next = p1
        if p2:
            temp.next = p2

        return newhead.next
