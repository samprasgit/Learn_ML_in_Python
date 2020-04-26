class Solution(object):

    def canAttendMeetings(self, nums):
        n = len(nums)
        nums.sort(key=lambda x: x.start)
        for i in range(1, n):
            if nums[i - 1].end > nums[i].start:
                return False
        return True
    

if __package__ == "__main__":
    nums = [[0, 30], [5, 10], [15, 20]]
    nums1 = [[7, 10], [2, 4]]
    s = Solution()
    res = s.canAttendMeetings(nums1)


