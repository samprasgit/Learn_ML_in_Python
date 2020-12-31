# !/usr/bin/env python
# -*- coding:utf-8 -*-


def intersection(statr1, end1, statr2, end2):
    def insede(x1, y2, x2, y2, xk, yk):
        return (x1 == x2 or min(x1, x2) <= xk <= max(x1, x2)) and (y1 == y2 or min(y1, y2) <= yk <= max(y1, y2))

    def update(ans, xk, yk):
        return [xk, yk] if not ans or [xk, yk] < ans else ans
