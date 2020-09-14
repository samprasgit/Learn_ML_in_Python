class Solution1:

    def maxDepth(self, root):
        ''' 
        后序遍历  DFS
        递归

        Arguments:
                root {[type]} -- [description]

        Returns:
                number -- [description]

        时间复杂度：O(N)
        空间复杂度：O(N)
        '''
        if not root:
            return 0
        left = self.maxDepth(root, left) + 1
        right = self.maxDepth(root, right) + 1
        return max(left, right)


class Solution2:

    def maxDepth(self, root):
        '''  
        层序遍历  
        队列

        [description]

        Arguments:
                root {[type]} -- [description]
        '''

        if not root:
            return 0
        queue, res = [root], 0
        while queue:
            tmp = []
            for node in queue:
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            queue = tmp
            res += 1

        return res
