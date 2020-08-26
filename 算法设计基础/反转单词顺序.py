class Solution:
    # 分词 反转

    def reverseWords(self, s):
        # 去除首尾空格
        s = s.strip()
        strs = s.split()  # 分词
        strs.reverse()
        # 拼接字符串并返回
        return " ".join(strs)
