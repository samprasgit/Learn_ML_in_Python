# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-21 13:56:41
# 描述: 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)


class TreeNode:

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        def recur(pRoot1, pRoot2):
            if not pRoot2:
                return True
            if not pRoot1 or pRoot1.val != pRoot2.val:
                return False

            return recur(pRoot1.left, pRoot2.left) and recur(pRoot1.right, pRoot2.right)

        return bool(pRoot1 and pRoot2) and (recur(pRoot1, pRoot2) or self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2))
