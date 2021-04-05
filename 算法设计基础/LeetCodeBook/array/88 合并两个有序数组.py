#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   88 合并两个有序数组.py
@Time    :   2021/04/05 04:33:35
@Author  :   Samprasgit 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
class Solution:
    def merge(self,nums1,num2):
        """
        双指针
        """
        # l为nums1的m开始索引 r为nums2的n开始索引 index为修改nums1的开始索引
        l,r,index=m-1,n-1,len(nums1)-1
        while l>=0  and r>=0 :
            if nums1[l]>=nums2[r]:
                nums1[index]=nums1[l]
                l-=1
            else:
                nums1[index]=nums2[r]
                r-=1 
            index-=1   

        #  nums2没有遍历完n个，则继续遍历，直到n个完成
        while r>=0:
            nums1[index]=nums2[r]
            r-=1
            index-=1




