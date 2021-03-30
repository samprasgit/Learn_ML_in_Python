#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/30 8:02 下午
# 搜索二维矩阵
class Solution:
    def searchMatrix(self,matrix,target):
        # 直接遍历
        M,N=len(matrix),len(mstrix[0])
        for i in  range(M):
            for j in range(N):
                if matrix[i][j]==target:
                    return True
        return False

