#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/2/22 3:33 下午


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def buildTree(self, preorder, inorder):
        """
        递归
        时间复杂度：O(N)
        空间复杂度：O(N)
        """

        def myBuilderTree(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:
                return None
            #   前序遍历的第一个节点就是根节点
            preorder_root = preorder_left
            # 中序遍历中定位根节点
            inorder_root = index[preorder[preorder_root]]

            #  先把根节点建立出来
            root = TreeNode(preorder[preorder_root])
            # 得到左子树的节点数目
            size_left_subtree = inorder_root - inorder_left
            #  递归的构造左子树 并连接到很节点
            root.left = myBuilderTree(preorder_left + 1, preorder_left + size_left_subtree, inorder_left,
                                      inorder_root - 1)

            # 递归的构造右子树
            root.right = myBuilderTree(preorder_left + size_left_subtree, preorder_right, inorder_root + 1,
                                       inorder_right)
            return root

        n = len(preorder)
        #  构造哈希映射，帮助我们快速定位根节点
        index = {element: i for i, element in enumerate(inorder)}
        return myBuilderTree(0, n - 1, 0, n - 1)




    def buildTree(self,preorder,inorder):
        """
        迭代
        时间复杂度：O(N)
        空间复杂度：O(N)
        """
        




