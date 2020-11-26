class Solution1:

    def maxmumGap(self, nums):
        """"

        直接排序
        时间复杂度：O(nlog(n))

        Arguments:
                nums {[type]} -- [description]
        """
        n = len(nums)
        if n < 2:
            return 0
        nums.sort()
        res = 0
        for i in range(1, n):
            res = max(res, nums[i] - nums[i - 1])

        return res

# O(N)时间复杂度 桶排序 基数排序
