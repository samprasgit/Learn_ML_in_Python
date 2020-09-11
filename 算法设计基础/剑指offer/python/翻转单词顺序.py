class Solution1:

    def reverseWords(self, s):
        '''

        双指针
        '''
        s = s.strip()
        i = j = len(s) - 1
        res = []
        while i >= 0:
            while i >= 0 and s[i] != ' ':
                i -= 1
            res.append(s[i + 1:j + 1])
            while s[j] != ' ':
                i -= 1
            j = i
        return ' '.join(res)
