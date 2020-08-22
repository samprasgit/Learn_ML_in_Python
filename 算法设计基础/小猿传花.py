def yuan(n, k):
    dp = []
    dp[0][0] = 1
    for i in range(k):
        for j in range(n):
            dp[i][j] = dp[(i - 1)][(j - 1 + n) % n] + \
                dp[(i - 1)][(j + 1 + n) % n]
    return dp[k][0]
