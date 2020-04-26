class TreeNode:

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def levelOrder(self, root):
        levels = []
        if not root:
            return False

        def helper(node, level):

            if len(levels) == level:
                levels.append([])

            levels[level].append(node.val)

            if node.left:
                helper(node.left, level + 1)
            if node.right:
                helper(node.right, level + 1)
        helper(root, 0)
        return levels
