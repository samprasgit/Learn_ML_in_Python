str = input()

s = ''
s1 = ''
res = set()  # set不包含重复的内容
for ele in str:
    if ele == s1:  # 当后一个元素等于前一个元素时，
        s = s + ele
    else:
        s = ele  # 罗列结果需要的字串
        s1 = ele  # 用于标记前一个元素
    res.add(s)

print(len(res))
