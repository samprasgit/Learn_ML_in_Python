def levelOrder(root):
    if not root:
        return []
    res = []
    curr = root
    queue = [curr]
    while queue:
        curr = queue.pop()
        res.append(curr.val)
        if curr.left:
            queue.append(curr.left)
        if curr.right:
            queue.append(curr.right)
    return res
