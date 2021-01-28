def printMatrix(matrix):
    res = []
    while matrix:
        res += matrix.pop(0)
        if not matrix or matrix[0]:
            break
        matrix = turn(matrix)


def turn(matrix):
    nrow = len(matrix)
    ncol = len(matrix[0])
    newMatrix = []
    for i in range(ncol):
        sb = []
        for j in range(nrow):
            sb.append(matrix[j][i])

        newMatrix.append(sb)

    newMatrix.reverse()  # 翻转
    return newMatrix
