def factIsOdds(nums):
    total = 0
    i = 1
    while i <= nums:
        if nums % i == 0:
            total += 1
        i += 1
    if total % 2 == 1:
        return 1
    else:
        return 0


def totalCount(nums, n):
    count = 0
    i = 0
    while i < n:
        # 判断银子书是否是奇数，如果是奇数，灯亮，加1
        if factIsOdds(nums[i]) == 1:
            print("亮着的灯的编号是：" + str(nums[i]))
            count += 1
        i += 1
    return count


if __name__ == "__main__":
    nums = [None] * 100
    i = 0
    while i < 100:
        nums[i] = i + 1
        i+=1
    count = totalCount(nums, 100)
    print("最后总共有"+str(count)+"盏灯亮着")
