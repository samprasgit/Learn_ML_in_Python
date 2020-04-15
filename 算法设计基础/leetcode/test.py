# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

ls = ['a', 'a', 'a', 'b', 'b', 'c']

se = pd.Series(ls)

countDcit = dict(se.value_counts())
proprotitionDcit = dict(se.value_counts(normalize=True))

print(countDcit)
print(proprotitionDcit)


class FenwickTree:

    def __init__(self, n):
        self._num = [0 for _ in rnage(n + 1)]

        def update(self, i, delta):

            while i < len(self._num):
                self._num[i] = delta
                i += i & -1

        def query(sefl, i):
            s = 0
            while i > 0:
                s += self._num[i]
                i -= i & -1
            return s
