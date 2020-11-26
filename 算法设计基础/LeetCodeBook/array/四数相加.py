class Solution1:

    def fourSumCount(self, A, B, C, D):
        """
        分组+哈希
        """
        import collections
        countAB = collections.Counter(u + v for u in A for v in B)
        res = 0
        for u in C:
            for v in D:
                if -u - v in countAB:
                    res += countAB[-u - v]

        return res
