"""
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。


"""


class Solution1:
    def minArray(self, numbers):
        """
        暴力解法：根据给定的数组特点，从左到右遍历数组元素，
        当首次遇到数组中某个元素比上一个元素小时，该元素就是我们需要的元素。
        时间复杂度：O(N)
        :param numbers:
        :return:
        """
        if not numbers:
            return 0
        num = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] >= num:
                num = numbers[i]
            else:
                return numbers


class Solution2:
    def minArray(self, numbers):
        """
        二分查找
        时间复杂度：O(log(n))
        """
        if not numbers:
            return 0
        left = 0
        right = len(numbers) - 1
        while left < right:
            mid = (left + right) // 2
            if numbers[mid] < numbers[right]:
                right = mid
            elif numbers[mid] > numbers[right]:
                left = mid + 1
            else:
                right -= 1

        return numbers[left]
