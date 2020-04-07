import random

    """等概率地从大小为n的数组中选取m个整数"""
def getRandom(nums, n, m):
    if nums == None or n < 0 or n < m:
        print("参数不合理！")
        return
    i = 0
    while i < m:
        j = random.randint(1, n - 1)
        # 随机选出的元素放到数组的前面
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp
        i += 1


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = 10
    m = 6
    getRandom(a, 10, 6)
    i = 0
    while i < m:
        print(a[i], )
        i += 1
