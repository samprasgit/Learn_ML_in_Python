## 1、罪犯转移

C市现在要转移一批罪犯到D市，C市有n名罪犯，按照入狱时间有顺序，另外每个罪犯有一个罪行值，值越大罪越重。现在为了方便管理，市长决定转移入狱时间连续的c名犯人，同时要求转移犯人的罪行值之和不超过t，问有多少种选择的方式（一组测试用例可能包含多组数据，请注意处理）？

输入：

```
3 100 2
1 2 3
```

输出：

```
2
```

```python
def func(n,t,c,nums):
    dp=sum([nums[i] for  i in range(c)])
    res=0 
    for i in  range(c,n):
        if dp<=t:
            res+=1 
        dp+=nums[i]-nums[i-c]
    if dp<=t:
        res+=1

    print(res)

        
while True:
    try:
        n,t,c=map(int,input().split())
        nums=list(map(int,input().split()))
        func(n,t,c,nums)
    except:
        break 
```

2蘑菇阵

现在有两个好友A和B，住在一片长有蘑菇的由n＊m个方格组成的草地，A在(1,1),B在(n,m)。现在A想要拜访B，由于她只想去B的家，所以每次她只会走(i,j+1)或(i+1,j)这样的路线，在草地上有k个蘑菇种在格子里(多个蘑菇可能在同一方格),问：A如果每一步随机选择的话(若她在边界上，则只有一种选择)，那么她不碰到蘑菇走到B的家的概率是多少？

```python
while True:
    try:
        (n, m, k) = (int(x) for x in input().split())
        matrix = [[0 for i in range(m)] for j in range(n)]
        dp = [[0.0 for i in range(m)] for j in range(n)]
        for i in range(k):
            (x, y) = (int(x) for x in input().split())
            matrix[x - 1][y - 1] = 1
        dp[0][0] = 1
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    if i < n - 1  and j < m - 1:
                        dp[i + 1][j] += 0.5 * dp[i][j]
                        dp[i][j + 1] += 0.5 * dp[i][j]
                    elif i == n - 1 and j < m - 1:
                        dp[i][j + 1] += dp[i][j]
                    elif i < n - 1:
                        dp[i + 1][j] += dp[i][j]
        print('%.2f'%(dp[-1][-1]))
    except:
        break
```

