# print("1500-2700之间可以同时整除5和7的数字：")
# for i in range(1500, 2700):
#     nums = i
#     if nums % 5 == 0 and nums % 7 == 0:
#         print(nums)


contest = input(
    "请依次输入v1，v2，t，s，l（空格隔开）其中(v1,v2< =100;t< =300;s< =10;l< =10000且为v1,v2的公倍数)").split()


'''
v1:兔子速度
v2:乌龟速度
t:兔子领先距离
s:兔子休息时间
l:赛道长度
'''

v1, v2, t, s, l = (int(i) for i in range(contest))
