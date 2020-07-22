class TreeNode:

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def preorderTraversal(self, root):
        '''前序遍历'''
        res = []

        def dfs(root):
            nonlocal res
            if not root:
                return

            res.append(root.val)
            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return res

    def inorderTraversal(self, root):
        '''中序遍历'''
        res = []

        def dfs(root):
            nonlocal res
            if not root:
                return

            dfs(root.left)
            res.append(root.val)
            dfs(root.right)

        dfs(root)
        return res

    def postorderTraversal(self, root):
        '''后序遍历'''
        res = []

        def dfs(root):
            nonlocal res
            if not root:
                return

            dfs(root.left)
            dfs(root.right)
            res.append(root.val)

        dfs(root)
        return res
