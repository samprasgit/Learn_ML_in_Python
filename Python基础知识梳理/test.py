s = 'amabc'
str_mapper = {
    "a": 0,
    "b": 1,
    "c": 2,
    "x": 3,
    "y": 4,
    "z": 5
}


def check(s, pre, l, r):
    for i in range(6):
        if s[l] in str_mapper and i == str_mapper[s[l]]:
            cnt = 1
        else:
            cnt = 0
        if (pre[r][i] - pre[l][i] + cnt) % 2 != 0:
            return False
    return True


def findTheLongestSubstring(s):
    n = len(s)

    pre = [[0] * 5 for _ in range(n)]

    # pre
    for i in range(n):
        for j in range(5):
            if s[i] in str_mapper and str_mapper[s[i]] == j:
                pre[i][j] = pre[i - 1][j] + 1
            else:
                pre[i][j] = pre[i - 1][j]
    for i in range(n - 1, -1, -1):
        for j in range(n - i):
            if check(s, pre, j, i + j):
                return i + 1
    return 0


print(s)
