class Solution(object):

    def threeSumSmaller(self, nums, target):

        n = len(nums)
        # 数组排序
        nums.sort()
        # 初始化符合条件的个数为0
        count = 0
        if n < 1:
            return 0
        # 三个数之和，所以要遍历到倒数第三个数
        for i in range(n - 2):
            # 最小指针是当前值下一位
            j = i + 1
            # 最大指针是数组最后一位
            k = n - 1

            while (j < k):
                # 三数之和小于目标值，只需计算最小指针到最大指针的距离，加到统计个数中
                if (nums[i] + nums[j] + nums[k] < target):
                    count += k - j
                    j += 1
                else:
                    k -= 1
        return count


if __name__ == "__main__":
    nums = [-2, 0, 1, 3]
    nums1 = [0, 0, 0]
    target = 2
    target1 = 0
    s = Solution()

    res = s.threeSumSmaller(nums, target)
    print(res)
