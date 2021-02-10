# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import collections


class Solution1:
    def characterReplacement(self, s, k):
        """
        滑动窗口
        """
        char_num = collections.defaultdict(int)
        if s is None or len(s) == 0:
            return 0
        left = 0
        res = 0
        for right in range(len(s)):
            char_num[s[right]] += 1
            res = max(res, char_num[s[right]])
            # right-left+1 当前窗口的长度
            if right - left + 1 > res + k:
                char_num[s[left]] -= 1
                left += 1
        return res
