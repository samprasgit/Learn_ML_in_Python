class Solution1:

    def trap(self, height):
        """

        暴力法，直接遍历
        时间复杂度：O(n^2)

        Arguments:
                height {[type]} -- [description]
        """
        ans = 0
        n = len(height)
        for i in range(n - 1):
            max_left, max_right = 0, 0
            for j in range(i, -1, -1):
                max_left = max(max_left, height[j])
                j -= 1
            for j in range(i, n):
                max_right = max(max_right, height[j])
                j += 1
            ans += min(max_left, max_right) - height[i]

        return ans


class Solution2:

    def trap(self, height):
        """

        动态规划
        Arguments:
                height {[type]} -- [description]

        Returns:
                number -- [description]
        """
        if not height:
            return 0
        ans = 0

        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        left_max[0] = height[0]
        for i in range(n):
            left_max[i] = max(height[i], left_max[i - 1])
        right_max[n - 1] = height[n - 1]
        for i in range(n - 2, -1, -1):
            right_max[i] = max(height[i], right_max[i + 1])

        for i in range(n - 1):
            ans += min(left_max[i], right_max[i]) - height[i]

        return ans


# class Solution4:

    #     def trap(self, height):
    #         """

    #         双指针

    #         Arguments:
    #                 height {[type]} -- [description]
    #         """

# class Solution4:

#     def trap(self, height):
#         """

#         单调栈

#         Arguments:
#                 height {[type]} -- [description]
#         """


s = Solution2()
height1 = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
height2 = [4, 2, 0, 3, 2, 5]
print(s.trap(height1))
