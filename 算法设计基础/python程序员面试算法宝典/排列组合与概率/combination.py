def combination(nums):
    """暴力破解法"""
    count = 1
    nums1 = nums        #1最多的个数
    nums2 = nums // 2   #2最多的个数
    nums3 = nums // 5   #5最多的个数

    x = 0
    while x <= nums1:
        y = 0
        while y <= nums2:
            z = 0
            while z <= nums3:
                if x + 2 * y + 5 * z == nums:
                    count += 1
                z += 1
            y += 1
        x += 1
    return count


def combinationCount(nums):
    """数字规律法"""

    m = 0
    count = 0
    while m <= nums:
        count += (m + 2) // 2
        m += 5
    return count


if __name__ == "__main__":
    res1 = combinationCount(100)
    print(res1)
