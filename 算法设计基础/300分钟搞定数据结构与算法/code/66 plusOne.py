# !/usr/bin/env python

# 从低位加1，若和等于10，向高位进一位，继续计算，否则返回该数组


def plusOne(digits):
        # 倒序遍历
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] <= 8:
            digits[i] += 1
            return digits
        else:
            digits[i] = 0

    return [1] + digits


print(plusOne([3, 9, 9]))
