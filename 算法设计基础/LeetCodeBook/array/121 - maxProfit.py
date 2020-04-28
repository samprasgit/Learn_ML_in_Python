class Solution:

    def maxProfit(self, prices):
        if len(prices) < 1:
            return 0
        dp = []
        dp.append(0)
        min_value = prices[0]
        for i in range(1, len(prices)):
            pd.append(max(dp[i - 1], prices[i] - min_value))
            if prices[i] < min_value:
                min_value = prices[i]

        return dp[-1]


if __name__ == "__main__":
    prices = [7, 1, 5, 3, 6]
    s = Solution()
    print(s.maxProfit(prices))
