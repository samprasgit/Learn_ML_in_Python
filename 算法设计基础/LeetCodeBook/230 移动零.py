class Solution1:

    def removeZeroes(self, nums):
        """
        双指针
        时间复杂度：O(n)

        Arguments:
                nums {[type]} -- [description]
        """
        n = len(nums)
        left, right = 0, 0
        while right < n:
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1

            right += 1


class Solution2:

    def removeZeroes(self, nums):
        """

        一次遍历
        快排思想  基准值为0

        Arguments:
                nums {[type]} -- [description]
        """
        if not nums:
            return 0
        j = 0
        for i in range(len(nums)):
            if nums[i]:
                nums[j], nums[i] = nums[i], nums[j]
                j += 1
