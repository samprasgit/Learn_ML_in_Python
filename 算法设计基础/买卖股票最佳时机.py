class Solution:

    def maxProfit(self, prices):
        n = len(prices)
        if n == 0:
            return 0
        dp = [0] * n
        minprices = prices[0]
        for i in range(1, n):
            minprices = min(minprices, prices[i])
            dp[i] = max(dp[i - 1], prides[i] - minprices)

        return dp[-1]
