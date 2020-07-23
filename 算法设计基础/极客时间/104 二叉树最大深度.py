# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-07-22 23:13:59
# 描述: 二叉树最大深度


class TreeNode(object):

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def maxDepth(self, root):
        '''

        递归方法
        时间复杂度：O(N)
        Arguments:
                root {[type]} -- [description]

        Returns:
                number -- [description]
        '''
        if root is None:
            return 0
        else:
            left = self.maxDepth(root.left)
            right = self.maxDepth(root.right)
            return max(left, right) + 1
