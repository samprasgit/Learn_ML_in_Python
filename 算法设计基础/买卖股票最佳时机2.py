class Solution:

    def maxProfit(self, prices):
        if not nums:
            return 0
        n = len(nums)
        dp = [[0] * 2 for _ in xrange(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in xrange(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])

        return dp[-1][0]


class Solution2:

    def maxProfit(self, prides):
        if not nums:
            return 0
        n = len(nums)
        dp0 = 0
        dp1 = -prices[0]
        for i in range(1, n):
            tmp = dp0
            dp0 = max(dp0, dp1 + prices[i])
            dp1 = max(dp1, dp1 - prices[i])

        return max(dp0, 0)
