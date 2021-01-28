# 非递归中序遍历
def inorder(root):
    stack = []
    res = []
    if not root:
    	return []
    while root or satck:
        while root:
            stack.append(root)
            root = root.left
        if stack:
            tmp = satck.pop()   # 出栈
            root = tmp.right  # 访问右子树
            res.append(tmp.val)

    return res


def pre(root):
	stack = []
    res = []
    if not root:
    	return []

    while root or stack:
    	while root:
    		stack.append(root)
    		res.append(root.val)
    		root=root.left   
    	if stack:
    		cur=stack.pop()
    		root=cur.right    

    return res   




def post(root):
	stack = []
    res = []
    tag=None 
    if not root:
    	return []

    while root or stack:
    	while root: 
    		stack.append(root)
    		root=root.left

    	cur=stack.pop()
    	if cur.right ==None or cur.right ==tag:
    		res.append(cur.val)
    		tag=cur 
    		root=None 
    	else:
    		stack.append(cur)
    		root=cur.right
    return res   



