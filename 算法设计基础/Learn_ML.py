nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]


def maxSubArray(nums):
    n = len(nums)
    # 确定终止分割的条件
    if n == 1:
        return nums[0]
    left = maxSubArray(nums[:n // 2])
    right = maxSubArray(nums[n // 2:])

    max_l = nums[n // 2 - 1]
    tmp = 0
    for i in range(n // 2 - 1, -1, -1):
        tmp += nums[i]
        max_l = max(tmp, max_l)

    max_r = nums[n // 2]
    tmp = 0
    for i in range(n // 2, n):
        tmp += nums[i]
        max_r = max(tmp, max_r)

    return max(left, right, max_l + max_r)


print(maxSubArray(nums))
