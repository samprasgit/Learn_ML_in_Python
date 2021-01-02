class Solution1:

    def myAtoi(self, s):
        """
        直接遍历
        """


class Solution2:
    def myAtoi(self, s):
        """
        正则表达式
        :param s:
        :return:
        """

        import re
        matches = re.match('[ ]*([+-]\d+)', s)
        if not matches:
            return 0
        res = int(matches.group(1))
        return max(min(res, 2 ** 31 - 1), -2 ** 31)
