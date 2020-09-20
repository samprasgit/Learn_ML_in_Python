while True:
    try:
        n, m = map(int, input().split())  # n 行 m列
        line = [[0] * m] * n  # 初始化二维数组
        for i in range(n):
            line[i] = input().split(" ")
            line[i] = [int(j) for j in line[i]]  # 输入二维数组，同行数字用空格分隔，不同行则用回车换行

        print(line)
    except:
        break
