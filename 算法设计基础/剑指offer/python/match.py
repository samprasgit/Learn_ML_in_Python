# !/usr/bin/env python
# -*- coding:utf-8 -*-
# time: 2020-05-21 16:15:13
# 描述: 正则表达式匹配


class Solution1:

    def match(sefl, s, pattern):
        '''递归

        [description]

        Arguments:
                sefl {[type]} -- [description]
                s {[type]} -- [description]
                pattern {[type]} -- [description]

        Returns:
                bool -- [description]
        '''
        if len(s) == 0 and len(pattern) == 0:
            return True
        if len(s) > 0 and len(pattern) == 0:
            return False

        # 如果模式第二个字符串是*
        if len(pattern) > 1 and pattern[1] == '*':
            if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
                # 如果第一个字符匹配，三种可能：
                # 1 字符串位移1位
                # 2 字符串位移1位，模式位移2位
                # 3 模式位移两位】
                return self.match(s, pattern[2:]) or self.match(s[1:], pattern)
            else:
                # 如果第一个字符不匹配，模式往后位移2位，相当于忽略x*
                return self.match(s, pattern[2:])
        # 如果模式第二个字符不是*
        if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
            return self.match(s[1:], pattern[2:])
        else:
            return False


class Solution2:

    def match(self, s, p):
        '''动态规划

        [description]

        Arguments:
                s {[type]} -- [description]
                pattern {[type]} -- [description]
        '''
        s, p = '#' + s, '#' + p
        m, n = len(s), len(p)
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        for i in range(m):
            for j in range(1, n):
                if i == 0:
                    dp[i][j] = j > 1 and p[j] == '*' and dp[i][j - 2]
                elif p[j] in [s[i], '.']:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j] == '*':
                    dp[i][j] = j > 1 and dp[i][
                        j - 2] or p[j - 1] in [s[i], '.'] and dp[i - 1][j]
                else:
                    dp[i][j] = False

        return dp[-1][-1]
