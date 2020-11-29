class Solution1:

    def largestPerimeter(self, A):
        """

        贪心+排序
        时间复杂度  O(nlog(n))
        """
        A.sort(reverse=True)
        n = len(A)
        for i in range(n - 2):
            if A[i + 2] + A[i + 1] > A[i]:
                return A[i + 2] + A[i + 1] + A[i]

        return 0


A = [3, 3, 2, 6]
s = Solution1()
print(s.largestPerimeter(A))
