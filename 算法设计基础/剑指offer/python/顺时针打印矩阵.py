class Solution:

    def spiralOrder(self, matrix):
        '''  

        模拟

        '''
        if not matrix:
            return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
                # 左到右
            for i in range(l, r + 1):
                res.append(matrix[t][i])
            t += 1
            if t > b:
                break
            # 上到下

            for i in range(t, b + 1):
                res.append(matrix[i][r])
            r -= 1
            if l > r:
                break
            # 右到左
            for i in range(r, l - 1, -1):
                res.append(matrix[b][i])
            b -= 1
            if t > b:
                break
            # 下到上
            for i in range(b, t - 1, -1):
                res.append(matrix[i][l])
            l += 1
            if l > r:
                break

        return res
