class Solution:

    def maxSlidigWindows(self, nums, k):
        deque = collections.deque()
        res = []
        n = len(nums)
        for i, j in zip(range(1 - k, n + 1 - k), range(n)):
            if i > 0 and deque[0] == nums[i - 1]:
                deque.popleft()

            while deque and deque[-1] < nums[j]:
                deque.pop()  # 保持队列递减

            deque.append(nums[j])
            if i >= 0:
                res.append(deque[0])

        return res


class Solution2:

    def maxSlidingWindow(self, nums, k):
        if not nums or k == 0:
            return []
        deque = collections.deque()
        n = len(nums)
        # 未形成窗口时
        for i in range(k):
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
        # 形成窗口后
        res = [deque[0]]
        for i in range(k, n):
            if deque[0] == nums[i - k]:
                deque.popleft()
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
            res.append(deque[0])
        return res
