
# # 使用递归实现
# # 回溯思想


# def permutations(nums, first, end):
#     if first == end:
#         print(nums)
#     else:
#         for index in range(first, end):
#             nums[index], nums[first] = nums[first], nums[index]
#             permutations(nums, first + 1, end)
#             nums[index], nums[first] = nums[first], nums[index]


nums = [1, 2, 3, 4]
# print(permutations(nums, 0, 4))


# 深度优先搜索实现全排列
visit = [True, True, True, True]
temp = ["" for i in range(0, len(nums))]


def permutations(nums, first):
        # 递归出口
    if first == len(nums):
        print(nums)
        return
    # 递归主体
    for index in range(0, len(nums)):
        if visit[index] == True:
            temp[first] = nums[index]
            visit[index] = False
            permutations(nums, first + 1)
            visit[index] = True


print(permutations(nums, 0))
