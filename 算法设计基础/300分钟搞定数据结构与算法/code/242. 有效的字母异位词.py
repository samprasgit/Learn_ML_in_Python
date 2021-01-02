# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-04-09 23:20:27
# 描述: 有效的字母异位词

class Solution1:
    def isAnagram(s, t):
        # 直接排序
        if len(s) != len(t):
            return False
        return sorted(s) == sorted(t)


class Solution2:
    def isAnagram(s, t):
        """
        字典
        """
        from collections import Counter
        return Counter(s) == Counter(t)


class Solution3:
    def isAnagram(s, t):
        if len(s) != len(t):
            return False
        dict1, dict2 = {}, {}
        for item in s:
            if item not in dict1:
                dict1[item] = 1
            else:
                dict1[item] += 1

        for item in t:
            if item not in dict1:
                dict2[item] = 1
            else:
                dict2[item] += 1

        if dict1 == dict2:
            return True
        else:
            return False


class Solution4:
    def isAnagram(s, t):
        if len(s) != len(t):
            return False
        set_temp = set(s)

        for i in set_temp:
            if s.count(i) != t.count(i):
                return False

        return True




if __name__ == "__main__":
    S=Solution4()
    s, t = "aa", "a"

    print(S.isAnagram(s, t))
