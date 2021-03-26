#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/26 10:57 上午
#  递归 + 迭代


class Solution:
    def deleteDuplicates(self, head):
        #  递归 -- 跳过连续相等的元素
        if not head or not head.next:
            return head
        if head.val != head.next.val:
            head.next = self.deleteDuplicates(head.next)
        else:
            move = head.next
            while move.next and head.val == move.next.val:
                move = move.next
            return self.deleteDuplicates(move)

        return head

    def deleteDuplicates2(self, head):
        # 递归 -- 删除下一个相等的元素
        if not head or not head.next:
            return head
        head.next = self.deleteDuplicates2(head.next)
        return head if head.val != head.next.val else head.next

    def deleteDuplicates3(self, head):
        # 迭代 一次遍历
        if not head: return None
        prev, cur = head, head.next
        while cur:
            if cur.val == prev.val:
                prev.next = cur.next
            else:
                prev = cur
            cur = cur.next

        return head


    
