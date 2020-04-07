"""
剑指offer数组查找
题目：
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。



思路：



第一种就是二次遍历，分别遍历数组的行和列，看看有没有目标值

第二种思路就是从数组的右上角开始遍历，因为是数据是从左往右，从上到下递增的，分为三种情况：

1.如果右上角那个值恰好是目标值，over

2.右上角那个值比目标值大，这个值所在列被pass掉，其实也就是指针（假设存在指针）左移

3.右上角那个值比目标值小，这个值所在行被pass掉，也就是指针下移


"""

class Solution(object):
    # array 二维列表
    def find(self,target,array):
        """直接遍历"""
        for i in range(len(array)):
            for j in range(len(array[0])):
                if (target==array[i][j]):
                    return True

        return False


    def Find(self, target, array):
        # write code here
        """指针法"""
        row = 0
        col = len(array[0])-1
        while (row <= len(array) - 1 and col >= 0):
            if (target == array[row][col]):
                return True
            elif (target > array[row][col]):
                row += 1
            else:
                col -= 1
        return False


if __name__ == "__main__":
    target = 57
    target0 =7
    array = [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
    s = Solution()
    res0=s.find(target0, array)
    res = s.Find(target, array)
    print(res)
    print(res0)
