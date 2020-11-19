class Solution1:

    def removeZeroes(self, nums):
        """
        双指针

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
