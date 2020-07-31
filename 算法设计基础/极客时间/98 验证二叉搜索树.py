# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-22 23:20:29
# 描述: 验证二叉搜索树


class TreeNode(object):

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def isValidBSTree(self, root):
        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False
            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, val, lower):
                return False
            return True
        return helper(root)
