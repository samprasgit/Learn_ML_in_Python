#!/usr/bin/env python
# -*- encoding: utf-8 -*-

class Solution1:
    def checkStraightLine(self, coordinates):
        # 模拟  计算斜率
        seen = set()

        for [x1, y1], [x2, y2] in zip(coordinates, coordinates[1:]):
            if x1 == x2:
                k = "INF"
            else:
                k = (y2 - y1) / (x2 - x1)

            seen.add(k)
            if len(seen) > 1:
                return False
        return True
