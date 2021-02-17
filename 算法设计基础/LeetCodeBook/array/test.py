def countMinSetp(n, m):
    # n鸡蛋熟 m楼层数
    if n < 1 or m < 1:
        return 0
    f = [[0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            f[i][j] = j  # 最坏的初始化步数
    for i in range(2, n + 1):
        for j in range(1, m + 1):
            for k in range(1, j):
                f[i][j] = min(1 + max(f[i - 1][k - 1], f[i][j - k]))

    return f[-1][-1]


print(countMinSetp(2, 100))
