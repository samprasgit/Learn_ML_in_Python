# ! /usr/bin/env python
# -*- coding: utf-8 -aaaaaaaasssss
#
# ''；；'；；；；''、、
#
#
#
#
# *-
import collections


class Solution:
    def frequencySort(self, s):
        return "".join(i * j for i, j in collections.Counter(s).most_common())


s = Solution()
ss = "tree"
res = s.frequencySort(ss)
print(res)
