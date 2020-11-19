class Solution1:

    def allCellsDistOrder(self, R, C, r0, c0):
        """[summary]

        直接排序
        时间复杂度：O(RClog(RC))


        Arguments:
                R {[type]} -- [description]
                C {[type]} -- [description]
                r0 {[type]} -- [description]
                c0 {[type]} -- [description]
        """
        res = [(i, j) for i in range(R) for j in range(C)]
        res.sort(key=lambda x: abs(x[0] - r0) + abs(x[1] - c0))
        return res


class Solution2:

    def allCellsDistOrder(self, R, C, r0, c0):
        """

        桶排序  

        时间复杂度：O(RC)
        线性时间复杂度

        Arguments:
                R {[type]} -- [description]
                C {[type]} -- [description]
                r0 {[type]} -- [description]
                c0 {[type]} -- [description]
        """
        import collections
        max_dist = max(r0, R - 1 - r0) + max(c0, C - 1 - c0)
        bucket = collections.defaultdict(list)
        dist = lambda r1, c1, r2, c2: abs(r1 - r2)
