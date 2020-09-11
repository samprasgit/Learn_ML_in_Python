class Solution1:

    def isMatch(self, s, p):
        '''

        DP   
        '''
        # 边界条件
        if not p:
            return not s
        if not s and len(p) == 1:
            return False

        m, n = len(s) + 1, len(p) + 1
        dp = [[False for _ in range(n)] for _ in range(m)]
        # 初始状态
        dp[0][0] = True
        dp[0][1] = False

        for c in range(2, n):
            j = c - 1
            if
