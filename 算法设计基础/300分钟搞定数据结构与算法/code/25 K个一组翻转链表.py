# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-11 00:03:32
# 描述: K 个一组翻转链表


class ListNode:
    def __init__(self, x):
        self.value = x
        self.next = None


class Solution:

    def reverseKGroups(self, head, k):
        # 定义一个哨兵节点
        sentery = ListNode(0)
        pre = sentery
        start = head
        flag = True
        while head:
            for i in range(k):
                if not head:
                    # 剩余节点数量小于K,跳出

                    flag = False
                    break

                head = head.next

            if not flag:
                break
            # 上次翻转后节点连接这次翻转后节点
            pre.next = self.reverse(start, head)
            # 连接这次翻转以后的正常节点
            start.next = head
            # 更新位置
            pre = start
            # 更新位置
            start = head

        return sentery.next

    def reverse(self, start, end):
        pre, cur, nexts = None, start, start
        # 三个指针记性局部翻转
        while cur != end:
            nexts = nexts.next
            # 箭头反指
            cur.next = pre
            # 更新pre位置
            pre = cur
            # 更新cur位置
            cur = nexts

        return pre
