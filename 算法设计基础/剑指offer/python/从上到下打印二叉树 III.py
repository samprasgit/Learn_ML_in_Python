class TreeNode:

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution1:
    ''' 
    层序遍历  + 双端队列  

    奇/偶层逻辑分开

    '''

    def levelOrder(self, root):
        if not root:
            return []
        res, deque = [], collections.deque()
        deque.append(root)
        while deque:
            temp = []
            # 打印奇数层
            for _ in range(len(deque)):
                # 左到右打印
                node = deque.popleft()
                temp.append(node.val)
                # 左到右加入下层节点
                if node.left:
                    deque.append(node.left)
                if node.right:
                    deque.append(node.right)

            res.append(temp)

            # 若为空，提前退出
            if not deque:
                break

            # 打印偶数层
            temp = []
            for _ in range(len(deque)):
                # 右到左 打印
                node = deque.pop()
                temp.append(node.val)
                # 右到左 加入下层节点
                if node.right:
                    deque.appendleft(node.right)
                if node.left:
                    deque.appendleft(node.left)

            res.append(temp)

        return res


class Solution2:
    ''' 
    层序遍历 + 倒序


    '''

    def levelOrder(self, root):
        if not root:
            return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            res.append(temp[::-1] if len(res) % 2 else temp)

        return res
