# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-22 11:56:47
# 描述: 二叉搜索树中第K小的元素


# Definition for a binary tree node.
class TreeNode:

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def kthSmallest(self, root, k):
        '''
        递归的方法
        时间复杂度：O(N)
        空间复杂度：O(N)
        '''

        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
        return inorder(root)[k - 1]

    def kthSmallest2(self, root, k):
        """
        迭代
        栈
        """
        stack = []

        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right
