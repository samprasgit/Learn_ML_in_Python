class Solution(object):
    def threeSumCloset(self, nums, target):
        nums.sort()
        n = len(nums)
        ans = nums[0] + nums[1] + nums[2]
        for i in range(n - 1):
            # 第一个指针
            j = i + 1
            # 第二个指针
            k = n - 1
            while (j < k):
                sum = nums[i] + nums[j] + nums[k]
                if (abs(sum - target) < abs(ans - target)):
                    ans = sum

                if (sum > target):
                    k -= 1
                elif (sum < target):
                    j += 1
                else:
                    return ans
        return ans


if __name__ == "__main__":
    nums = [-1, 2, 1, -4]
    target = 1
    s = Solution()
    res = s.threeSumCloset(nums, target)
    print(res)
