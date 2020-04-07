"""
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。


"""

class Solution(object):
    def minNumberInRotateArray0(self,rotateArray):
        """
        暴力解法：根据给定的数组特点，从左到右遍历数组元素，
        当首次遇到数组中某个元素比上一个元素小时，该元素就是我们需要的元素。
        时间复杂度：O(N)
        :param rotateArray:
        :return:
        """
        if not rotateArray:
            return 0
        num=rotateArray[0]
        for i in range(1,len(rotateArray)):
            if rotateArray[i]>=num:
                num=rotateArray[i]
            else:
                return rotateArray[i]
