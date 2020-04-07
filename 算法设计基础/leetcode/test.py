# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

ls = ['a', 'a', 'a', 'b', 'b', 'c']

se = pd.Series(ls)

countDcit = dict(se.value_counts())
proprotitionDcit = dict(se.value_counts(normalize=True))

print(countDcit)
print(proprotitionDcit)
