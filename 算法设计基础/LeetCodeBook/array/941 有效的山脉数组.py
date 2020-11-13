class Solution1:
    def validMountainArray(self, A):
        # 线性扫描
        n = len(A)
        i = 0

        # 递增扫描
        while i + 1 < n and A[i] < A[i + 1]:
            i += 1

        if i == 0 or i == n - 1:
            return False

        while i + 1 < n and A[i] > A[i + 1]:
            i += 1

        return i == n - 1


class Solution2:
    def validMountainArray(self, A):
        # 双指针
        n = len(A)
        left, right = 0, len(A) - 1
        while left + 1 < n and A[left] < A[left + 1]:
            left += 1
        while right > 0 and A[right - 1] > A[right]:
            right -= 1

        return (left > 0 and right < n - 1 and left == right)
