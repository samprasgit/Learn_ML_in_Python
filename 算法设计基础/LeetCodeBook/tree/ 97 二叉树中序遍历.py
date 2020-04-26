

# 打印中序遍历-倒序
def dfs(root):
    if not root:
        return
    dfs(root.right)  # 右
    print(root.val)  # 根
    dfs(root.left)  # 左
